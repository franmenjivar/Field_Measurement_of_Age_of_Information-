"""IDF-DS fixed-wing UAS telemetry benchmark adapter (SpeedyBee / INAV subset).

Real autonomous flights over a 3 981 m circuit.  Two streams are available per
lap and they serve different purposes:

``processed/telemetry/LapNNN.csv``
    ~62 Hz INAV blackbox.  ``navPos``/``navVel`` are the estimator's position
    and velocity in a local NEU frame (centimetres and cm/s).  **This is the
    reference trajectory** for the downsampling study -- at 62 Hz it resolves
    the 100-200 ms information-age regime that a 1 Hz dataset cannot.

``processed/GPS/LapNNN_gps.csv``
    exactly 5.00 Hz raw GPS: geodetic position plus ``GPS_velned`` in cm/s.
    Used to georeference the lap and as an independent cross-check.

Unit conversions applied here (verified against Lap001): navPos cm -> m,
navVel cm/s -> m/s, attitude decidegrees -> degrees, NEU -> ENU.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.geodesy import GeodeticFrame, ned_to_enu
from ..core.provenance import Provenance
from ..core.schema import coerce_trajectory_frame, derive_kinematics
from .base import DatasetNotAvailable, TrajectoryDataset

_LAP_RE = re.compile(r"Lap(\d+)")

_NAV_COLUMNS = [
    "time (us)",
    "navPos[0]",
    "navPos[1]",
    "navPos[2]",
    "navVel[0]",
    "navVel[1]",
    "navVel[2]",
    "attitude[0]",
    "attitude[1]",
    "attitude[2]",
]


class IdfDsDataset(TrajectoryDataset):
    """One lap = one scenario with a single aircraft."""

    name = "idf_ds"

    def __init__(
        self,
        root: str | Path = "data/interim/idf_ds/Speedybee DataSet",
        *,
        source: str = "nav",
    ) -> None:
        """
        Parameters
        ----------
        source
            ``"nav"`` (default) uses the ~62 Hz estimator stream as the
            reference trajectory; ``"gps"`` uses the 5 Hz raw GPS instead.
        """
        super().__init__(root)
        if source not in {"nav", "gps"}:
            raise ValueError("source must be 'nav' or 'gps'")
        self.source = source
        self.tel_dir = self.root / "processed" / "telemetry"
        self.gps_dir = self.root / "processed" / "GPS"
        if not self.tel_dir.is_dir() or not self.gps_dir.is_dir():
            raise DatasetNotAvailable(
                f"idf_ds: expected processed/telemetry and processed/GPS under {self.root}"
            )
        self._frames: dict[str, GeodeticFrame] = {}
        self._rates: dict[str, float] = {}

    # ------------------------------------------------------------------ scenarios

    def scenarios(self) -> list[str]:
        """Laps that have both a telemetry and a GPS file."""
        tel = {m.group(0) for f in self.tel_dir.glob("Lap*.csv") if (m := _LAP_RE.match(f.stem))}
        gps = {m.group(0) for f in self.gps_dir.glob("Lap*_gps.csv") if (m := _LAP_RE.match(f.stem))}
        return sorted(tel & gps)

    def _paths(self, scenario_id: str) -> tuple[Path, Path]:
        tel = self.tel_dir / f"{scenario_id}.csv"
        gps = self.gps_dir / f"{scenario_id}_gps.csv"
        for p in (tel, gps):
            if not p.is_file():
                raise DatasetNotAvailable(f"idf_ds: missing {p}")
        return tel, gps

    # ------------------------------------------------------------------- loading

    def load_gps(self, scenario_id: str) -> pd.DataFrame:
        """Raw 5 Hz GPS stream for one lap, in the unified schema."""
        _, gps_path = self._paths(scenario_id)
        g = pd.read_csv(gps_path)
        t = g["time (us)"].to_numpy(dtype=float) / 1e6
        t -= t[0]

        lat = g["GPS_coord[0]"].to_numpy(dtype=float)
        lon = g["GPS_coord[1]"].to_numpy(dtype=float)
        alt = g["GPS_altitude"].to_numpy(dtype=float)

        frame = self._frame_for(scenario_id, lat, lon, alt)
        x, y, z = frame.to_enu(lat, lon, alt)

        # GPS_velned is [North, East, Down] in cm/s.
        vn = g["GPS_velned[0]"].to_numpy(dtype=float) / 100.0
        ve = g["GPS_velned[1]"].to_numpy(dtype=float) / 100.0
        vd = g["GPS_velned[2]"].to_numpy(dtype=float) / 100.0
        vx, vy, vz = ned_to_enu(vn, ve, vd)

        df = pd.DataFrame(
            {
                "dataset": self.name,
                "scenario_id": scenario_id,
                "uas_id": f"{scenario_id}-A",
                "t_s": t,
                "lat_deg": lat,
                "lon_deg": lon,
                "alt_m_msl": alt,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_mps": vx,
                "vy_mps": vy,
                "vz_mps": vz,
                "speed_mps": g["GPS_speed (m/s)"].to_numpy(dtype=float),
                "heading_deg": g["GPS_ground_course"].to_numpy(dtype=float) % 360.0,
                "source_type": "ground_truth",
                "quality": g.get("GPS_hdop", pd.Series(np.nan, index=g.index)),
            }
        )
        return self._finalise(df)

    def load_scenario(self, scenario_id: str) -> pd.DataFrame:
        """Reference trajectory for one lap (62 Hz nav stream by default)."""
        if self.source == "gps":
            return self.load_gps(scenario_id)

        tel_path, gps_path = self._paths(scenario_id)
        d = pd.read_csv(tel_path, usecols=lambda c: c in _NAV_COLUMNS)
        missing = [c for c in _NAV_COLUMNS if c not in d.columns]
        if missing:
            raise DatasetNotAvailable(f"idf_ds: {tel_path.name} lacks {missing}")

        t = d["time (us)"].to_numpy(dtype=float) / 1e6
        t -= t[0]

        # navPos/navVel are NEU-ordered in INAV: [North, East, Up].
        n_m = d["navPos[0]"].to_numpy(dtype=float) / 100.0
        e_m = d["navPos[1]"].to_numpy(dtype=float) / 100.0
        u_m = d["navPos[2]"].to_numpy(dtype=float) / 100.0
        vn = d["navVel[0]"].to_numpy(dtype=float) / 100.0
        ve = d["navVel[1]"].to_numpy(dtype=float) / 100.0
        vu = d["navVel[2]"].to_numpy(dtype=float) / 100.0

        # Georeference the local nav frame using the lap's GPS home position, so
        # laps can later be placed in a shared frame for constructed encounters.
        g = pd.read_csv(gps_path, usecols=["GPS_home_lat", "GPS_home_lon", "GPS_altitude"])
        home_lat = float(g["GPS_home_lat"].dropna().median())
        home_lon = float(g["GPS_home_lon"].dropna().median())
        home_alt = float(g["GPS_altitude"].dropna().median())
        frame = GeodeticFrame(home_lat, home_lon, home_alt)
        self._frames[scenario_id] = frame
        lat, lon, alt = frame.to_geodetic(e_m, n_m, u_m)

        df = pd.DataFrame(
            {
                "dataset": self.name,
                "scenario_id": scenario_id,
                "uas_id": f"{scenario_id}-A",
                "t_s": t,
                "lat_deg": lat,
                "lon_deg": lon,
                "alt_m_msl": alt,
                "x_m": e_m,
                "y_m": n_m,
                "z_m": u_m,
                "vx_mps": ve,
                "vy_mps": vn,
                "vz_mps": vu,
                "speed_mps": np.hypot(ve, vn),
                "heading_deg": np.degrees(np.arctan2(ve, vn)) % 360.0,
                "source_type": "ground_truth",
                "quality": np.nan,
            }
        )
        # attitude[2] is yaw in decidegrees; kept for maneuver segmentation.
        df["yaw_deg"] = d["attitude[2]"].to_numpy(dtype=float) / 10.0
        df["roll_deg"] = d["attitude[0]"].to_numpy(dtype=float) / 10.0
        df["pitch_deg"] = d["attitude[1]"].to_numpy(dtype=float) / 10.0
        return self._finalise(df)

    # ------------------------------------------------------------------- helpers

    def _frame_for(self, scenario_id, lat, lon, alt) -> GeodeticFrame:
        if scenario_id not in self._frames:
            self._frames[scenario_id] = GeodeticFrame.from_points(lat, lon, alt)
        return self._frames[scenario_id]

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        aux = [c for c in ("yaw_deg", "roll_deg", "pitch_deg") if c in df.columns]
        df = df.dropna(subset=["t_s", "x_m", "y_m"]).sort_values("t_s")
        df = df.drop_duplicates(subset=["uas_id", "t_s"], keep="first").reset_index(drop=True)
        extra = df[aux] if aux else None
        out = coerce_trajectory_frame(df)
        out = derive_kinematics(out)
        if extra is not None:
            out[aux] = extra
        sid = str(out["scenario_id"].iloc[0])
        dt = np.diff(out["t_s"].to_numpy())
        self._rates[sid] = float(1.0 / np.median(dt)) if len(dt) else np.nan
        return out

    def reference_rate_hz(self, scenario_id: str) -> float:
        if scenario_id not in self._rates:
            self.load_scenario(scenario_id)
        return self._rates[scenario_id]

    def frame(self, scenario_id: str) -> GeodeticFrame:
        if scenario_id not in self._frames:
            self.load_scenario(scenario_id)
        return self._frames[scenario_id]

    def provenance(self, scenario_id: str) -> Provenance:
        tel, gps = self._paths(scenario_id)
        p = Provenance(
            source_dataset="IDF_DS",
            scenario_id=f"SpeedyBee/{scenario_id}",
            reference_rate_hz=self.reference_rate_hz(scenario_id),
            coordinate_frame="ENU",
            origin=self.frame(scenario_id).as_dict(),
            notes=[
                f"source stream = {self.source}",
                "navPos cm -> m, navVel cm/s -> m/s, INAV NEU -> ENU",
                "local nav frame georeferenced via GPS_home_lat/lon",
            ],
        )
        p.add_source(str(tel if self.source == "nav" else gps))
        p.add_source(str(gps))
        return p
