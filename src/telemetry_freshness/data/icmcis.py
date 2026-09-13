"""ICMCIS drone detection & tracking dataset adapter.

External validation with an *independent* surveillance technology.  Where
BUBBLES observes aircraft through cooperative telemetry, here the observer is
radar, so the truth-versus-observation comparison does not share a failure mode
with the cooperative link.

Per scenario directory:

``YYYY-MM-DD_HH-MM-SS_v2.csv``
    the drone's own onboard log -- **ground truth**.  Carries geodetic position,
    ``speed(mps)``, ``velocityX/Y/Z(mps)``, attitude and an epoch ``timestamp``.
    A scenario may contain more than one, i.e. more than one aircraft.

``ALVIRA_scenario.csv`` / ``ARCUS_scenario.csv``
    radar tracks -- **observed**.  Position plus velocity as
    azimuth/elevation/speed, with the sensor's own track timestamp.

``DIANA_scenario.csv`` / ``VENUS_scenario.csv``
    RF detection and direction finding.  Bearing-only, so they cannot be turned
    into a position track and are exposed raw rather than mapped into the
    trajectory schema.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.geodesy import GeodeticFrame
from ..core.provenance import Provenance
from ..core.schema import coerce_trajectory_frame, derive_kinematics
from .base import DatasetNotAvailable, ObservationDataset, TrajectoryDataset

_DRONE_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

RADAR_SENSORS = ("ALVIRA", "ARCUS")
RF_SENSORS = ("DIANA", "VENUS")


class IcmcisDataset(TrajectoryDataset, ObservationDataset):
    """Onboard logs as truth, radar tracks as observation."""

    name = "icmcis"

    def __init__(
        self,
        root: str | Path = "data/raw/icmcis/drone_detection",
        *,
        splits: tuple[str, ...] = ("train", "test"),
    ) -> None:
        super().__init__(root)
        self.splits = splits
        self._dirs: dict[str, Path] = {}
        self._frames: dict[str, GeodeticFrame] = {}
        self._t0: dict[str, float] = {}
        self._index()

    def _index(self) -> None:
        for split in self.splits:
            # The archive nests the split one level deeper (train/train/...).
            for base in (self.root / split / split, self.root / split):
                if base.is_dir():
                    for d in sorted(p for p in base.iterdir() if p.is_dir()):
                        self._dirs[f"{split}/{d.name}"] = d
                    break
        if not self._dirs:
            raise DatasetNotAvailable(f"icmcis: no scenario directories under {self.root}")

    def scenarios(self) -> list[str]:
        return sorted(self._dirs)

    def _dir(self, scenario_id: str) -> Path:
        if scenario_id not in self._dirs:
            raise DatasetNotAvailable(f"icmcis: unknown scenario {scenario_id!r}")
        return self._dirs[scenario_id]

    def drone_logs(self, scenario_id: str) -> list[Path]:
        return sorted(
            p for p in self._dir(scenario_id).glob("*.csv") if _DRONE_LOG_RE.match(p.stem)
        )

    # ------------------------------------------------------------- truth

    def load_scenario(self, scenario_id: str) -> pd.DataFrame:
        """Onboard drone logs for one scenario, in the unified schema."""
        logs = self.drone_logs(scenario_id)
        if not logs:
            raise DatasetNotAvailable(f"icmcis: no drone log in {self._dir(scenario_id)}")

        parts = []
        for i, path in enumerate(logs):
            d = pd.read_csv(path, low_memory=False)
            need = {"latitude", "longitude", "altitude(m)", "timestamp"}
            if not need <= set(d.columns):
                continue
            d = d.dropna(subset=["latitude", "longitude", "timestamp"])
            # Logs occasionally repeat a timestamp; the store needs strict order.
            d = d.drop_duplicates(subset=["timestamp"], keep="first")
            d = d[(d["latitude"].abs() > 1e-6) & (d["longitude"].abs() > 1e-6)]
            if d.empty:
                continue
            d["_uas"] = f"{scenario_id.split('/')[-1]}-D{i + 1}"
            d["_t_epoch_s"] = d["timestamp"].astype(float) / 1000.0
            d["_src"] = path.name
            parts.append(d)

        if not parts:
            raise DatasetNotAvailable(f"icmcis: no usable drone rows in {scenario_id}")

        raw = pd.concat(parts, ignore_index=True)
        t0 = float(raw["_t_epoch_s"].min())
        self._t0[scenario_id] = t0

        frame = GeodeticFrame.from_points(
            raw["latitude"].to_numpy(float),
            raw["longitude"].to_numpy(float),
            raw["altitude(m)"].to_numpy(float),
        )
        self._frames[scenario_id] = frame
        x, y, z = frame.to_enu(
            raw["latitude"].to_numpy(float),
            raw["longitude"].to_numpy(float),
            raw["altitude(m)"].to_numpy(float),
        )

        # The log's velocityX/Y/Z is body/NED-flavoured and inconsistently
        # signed across scenarios, so heading is taken from yaw and combined
        # with the scalar ground speed instead.
        speed = pd.to_numeric(raw.get("speed(mps)"), errors="coerce").to_numpy(float)
        yaw = pd.to_numeric(raw.get("yaw(deg)"), errors="coerce").to_numpy(float)
        th = np.radians(yaw)
        vx, vy = speed * np.sin(th), speed * np.cos(th)

        df = pd.DataFrame(
            {
                "dataset": self.name,
                "scenario_id": scenario_id,
                "uas_id": raw["_uas"],
                "t_s": raw["_t_epoch_s"] - t0,
                "lat_deg": raw["latitude"].astype(float),
                "lon_deg": raw["longitude"].astype(float),
                "alt_m_msl": raw["altitude(m)"].astype(float),
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_mps": vx,
                "vy_mps": vy,
                "vz_mps": pd.to_numeric(raw.get("velocityZ(mps)"), errors="coerce"),
                "speed_mps": speed,
                "heading_deg": yaw % 360.0,
                "source_type": "ground_truth",
                "quality": pd.to_numeric(raw.get("satellites"), errors="coerce"),
            }
        )
        df["source_file"] = raw["_src"].to_numpy()
        df = df.sort_values(["uas_id", "t_s"])
        df = df.drop_duplicates(subset=["uas_id", "t_s"], keep="first").reset_index(drop=True)
        src = df.pop("source_file")
        out = derive_kinematics(coerce_trajectory_frame(df))
        out["source_file"] = src.to_numpy()
        return out

    # ---------------------------------------------------------- observed

    def load_observations(
        self, scenario_id: str, *, sensor: str = "ALVIRA"
    ) -> pd.DataFrame:
        """Radar tracks for one scenario, with the sensor's own timestamps.

        ``t_gen_s`` is the sensor's per-track timestamp and ``t_s`` the row's
        record time, so their difference is the observation age this sensor
        actually delivered.
        """
        sensor = sensor.upper()
        if sensor not in RADAR_SENSORS:
            raise ValueError(f"sensor must be one of {RADAR_SENSORS}, got {sensor!r}")
        path = self._dir(scenario_id) / f"{sensor}_scenario.csv"
        if not path.is_file():
            raise DatasetNotAvailable(f"icmcis: missing {path}")

        d = pd.read_csv(path, low_memory=False)
        p = sensor.capitalize()
        cols = {
            "track_ts": f"{p}TracksTrack_Timestamp",
            "track_id": f"{p}TracksTrack_id",
            "lat": f"{p}TracksTrackPosition_Latitude",
            "lon": f"{p}TracksTrackPosition_Longitude",
            "alt": f"{p}TracksTrackPosition_Altitude",
            "az": f"{p}TracksTrackVelocity_Azimuth",
            "spd": f"{p}TracksTrackVelocity_Speed",
        }
        missing = [c for c in cols.values() if c not in d.columns]
        if missing:
            raise DatasetNotAvailable(f"icmcis: {path.name} lacks {missing}")

        d = d.dropna(subset=[cols["lat"], cols["lon"]]).copy()
        if d.empty:
            return pd.DataFrame()

        # Ground truth first: it fixes the frame and the time origin.
        if scenario_id not in self._frames:
            self.load_scenario(scenario_id)
        frame = self._frames[scenario_id]
        t0 = self._t0[scenario_id]

        rec_s = pd.to_numeric(d["total in seconds"], errors="coerce").astype(float)
        gen_s = pd.to_datetime(d[cols["track_ts"]], errors="coerce", utc=True)
        # Explicit epoch difference: the integer resolution of a datetime64
        # series is pandas-version dependent, so never divide astype("int64").
        gen_epoch = (gen_s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()

        lat = d[cols["lat"]].astype(float).to_numpy()
        lon = d[cols["lon"]].astype(float).to_numpy()
        alt = pd.to_numeric(d[cols["alt"]], errors="coerce").fillna(0.0).to_numpy()
        x, y, z = frame.to_enu(lat, lon, alt)

        spd = pd.to_numeric(d[cols["spd"]], errors="coerce").to_numpy()
        az = np.radians(pd.to_numeric(d[cols["az"]], errors="coerce").to_numpy())

        out = pd.DataFrame(
            {
                "dataset": self.name,
                "scenario_id": scenario_id,
                "sensor": sensor,
                "track_id": d[cols["track_id"]].astype(str).to_numpy(),
                "t_s": rec_s.to_numpy() - t0,
                "t_gen_s": gen_epoch.to_numpy() - t0,
                "lat_deg": lat,
                "lon_deg": lon,
                "alt_m_msl": alt,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_mps": spd * np.sin(az),
                "vy_mps": spd * np.cos(az),
                "speed_mps": spd,
                "heading_deg": np.degrees(az) % 360.0,
                "source_type": "sensor_track",
            }
        )
        out = out.sort_values("t_gen_s").reset_index(drop=True)
        # The record clock ("total in seconds") is integer seconds while the
        # sensor timestamp has millisecond resolution, so this age is quantised
        # to ~1 s and is a coarse measure -- report it as such.
        out["observation_age_s"] = out["t_s"] - out["t_gen_s"]
        # Track update interval keeps the sensor's full millisecond resolution
        # and is the better measure of how often this radar refreshed a track.
        out["track_interval_s"] = out["t_gen_s"].diff()
        return out

    def load_rf(self, scenario_id: str, sensor: str = "DIANA") -> pd.DataFrame:
        """Raw RF sensor rows (bearing-only; not a position track)."""
        sensor = sensor.upper()
        if sensor not in RF_SENSORS:
            raise ValueError(f"sensor must be one of {RF_SENSORS}")
        path = self._dir(scenario_id) / f"{sensor}_scenario.csv"
        if not path.is_file():
            raise DatasetNotAvailable(f"icmcis: missing {path}")
        return pd.read_csv(path, low_memory=False)

    # -------------------------------------------------------------- misc

    def frame(self, scenario_id: str) -> GeodeticFrame:
        if scenario_id not in self._frames:
            self.load_scenario(scenario_id)
        return self._frames[scenario_id]

    def provenance(self, scenario_id: str) -> Provenance:
        p = Provenance(
            source_dataset="ICMCIS",
            scenario_id=scenario_id,
            coordinate_frame="ENU",
            origin=self.frame(scenario_id).as_dict(),
            notes=[
                "ground truth = onboard drone log; observed = radar track",
                "heading derived from yaw(deg) + speed(mps); the log's "
                "velocityX/Y/Z sign convention is inconsistent across scenarios",
                "DIANA/VENUS are bearing-only and are not mapped to trajectories",
            ],
        )
        for f in self.drone_logs(scenario_id):
            p.add_source(str(f))
        for s in RADAR_SENSORS:
            q = self._dir(scenario_id) / f"{s}_scenario.csv"
            if q.is_file():
                p.add_source(str(q))
        return p
