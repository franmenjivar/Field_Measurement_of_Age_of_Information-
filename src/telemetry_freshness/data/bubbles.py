"""BUBBLES U-space validation dataset adapter.

Real U-space campaign near Valencia with 14 drones airborne together.  The
dataset is unusual and valuable because it records **both** layers this study
needs, with their own clocks:

``input/telemetry/<drone>/logNNN.txt``
    one JSON message per line, carrying the UTM reception time (``time_ms``)
    *and* the onboard generation time (``message.time``).  Their difference is
    a measured one-way telemetry latency.

``utm_tracks/logNNN.txt``
    one JSON UTM surveillance cycle per line, each listing every tracked
    aircraft with ``t`` (cycle time) and ``t_last`` (time that aircraft's state
    was last refreshed).  ``t - t_last`` is a **measured Age of Information**.

Quirks handled here, all verified against the files (docs/DATASETS.md, BUBBLES):

* the ``utm_tracks`` clock runs ~+7200 s ahead of ``time_ms`` (local CEST
  written as if it were epoch); the offset is estimated per day, not assumed;
* ``message.time`` is epoch-ms for some airframes and a monotonic uptime
  counter for others -- the two are separated and labelled, never mixed;
* a small fraction of track lines carry a dict instead of a list.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.geodesy import GeodeticFrame, track_angle_to_enu_velocity
from ..core.provenance import Provenance
from ..core.schema import coerce_trajectory_frame, derive_kinematics
from .base import DatasetNotAvailable, ObservationDataset, TrajectoryDataset

#: ``message.time`` above this is an epoch timestamp in ms, below it an uptime counter.
_EPOCH_MS_THRESHOLD = 1e11

#: Rows whose reported latency exceeds this are treated as clock artefacts, not transport.
_MAX_PLAUSIBLE_LATENCY_S = 60.0

#: Observed tracks further than this from the campaign centroid are probe artefacts.
_PROBE_DISTANCE_M = 10_000.0


def _iter_json_lines(path: Path):
    """Yield parsed JSON objects, skipping malformed lines."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class BubblesDataset(TrajectoryDataset, ObservationDataset):
    """Ground-truth telemetry *and* UTM observed tracks from BUBBLES."""

    name = "bubbles"

    def __init__(
        self,
        root: str | Path = "data/raw/bubbles/DroneFlightDataset",
        *,
        include_probe_messages: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        include_probe_messages
            The campaign logs contain injected system-test messages mixed in
            with real telemetry.  They are identifiable by a placeholder
            payload (constant attitude 4/4/4, ``GS`` exactly 20,
            ``pilot_alerts`` as the string ``"none"``), an uptime rather than
            epoch generation clock, and a position ~40 km NW of the campaign
            site.  They are excluded by default; set this to keep them for
            inspection.  They are never real aircraft state and must not enter
            any trajectory or conflict analysis.
        """
        super().__init__(root)
        self.include_probe_messages = include_probe_messages
        self._cache: dict[str, pd.DataFrame] = {}
        self._obs_cache: dict[str, pd.DataFrame] = {}
        self._frames: dict[str, GeodeticFrame] = {}
        self._t0: dict[str, float] = {}
        self._offsets: dict[str, float] = {}
        self._stats: dict[str, dict] = {}

    # ------------------------------------------------------------------ scenarios

    def scenarios(self) -> list[str]:
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("20")
        )

    def _day_dir(self, scenario_id: str) -> Path:
        d = self.root / scenario_id
        if not d.is_dir():
            raise DatasetNotAvailable(f"bubbles: no scenario {scenario_id!r} under {self.root}")
        return d

    # ------------------------------------------------------------- ground truth

    def load_scenario(self, scenario_id: str) -> pd.DataFrame:
        """Onboard telemetry of every drone for one campaign day.

        Treated as the ground-truth layer: it is the aircraft's own reported
        state, timestamped at generation.
        """
        if scenario_id in self._cache:
            return self._cache[scenario_id]

        tel_root = self._day_dir(scenario_id) / "input" / "telemetry"
        if not tel_root.is_dir():
            raise DatasetNotAvailable(f"bubbles: missing {tel_root}")

        rows: list[dict] = []
        clock_kind: Counter = Counter()

        for drone_dir in sorted(p for p in tel_root.iterdir() if p.is_dir()):
            drone = drone_dir.name
            for f in sorted(drone_dir.glob("*.txt")):
                # message.time is epoch-ms for some airframes and an uptime
                # counter for others, and a single file can contain both, so the
                # unit is decided per record rather than per file.
                recs = list(_iter_json_lines(f))

                # Uptime counters are aligned to the reception clock so relative
                # spacing (jitter) survives while absolute latency does not.
                # Computed unconditionally: a mostly-epoch file can still contain
                # a handful of uptime records, and those need the offset too.
                deltas = [
                    r["time_ms"] - r["message"]["time"]
                    for r in recs
                    if isinstance(r.get("message"), dict)
                    and isinstance(r["message"].get("time"), int | float)
                    and r["message"]["time"] <= _EPOCH_MS_THRESHOLD
                ]
                offset_ms = float(np.median(deltas)) if deltas else np.nan

                for r in recs:
                    msg = r.get("message")
                    if not isinstance(msg, dict):
                        continue
                    pos = msg.get("position") or {}
                    spd = msg.get("speed") or {}
                    att = msg.get("attitude") or {}
                    lat, lon = pos.get("latitude"), pos.get("longitude")
                    if lat is None or lon is None:
                        continue

                    gen = msg.get("time")
                    if isinstance(gen, int | float):
                        if gen > _EPOCH_MS_THRESHOLD:
                            t_gen_ms, kind = float(gen), "epoch"
                        else:
                            t_gen_ms, kind = float(gen) + offset_ms, "uptime_aligned"
                    else:
                        t_gen_ms, kind = np.nan, "missing"
                    clock_kind[(drone, kind)] += 1

                    # Injected system-test message rather than aircraft state.
                    is_probe = (
                        isinstance(msg.get("pilot_alerts"), str)
                        or kind == "uptime_aligned"
                    )
                    if is_probe and not self.include_probe_messages:
                        clock_kind[(drone, "probe_dropped")] += 1
                        continue

                    rows.append(
                        {
                            "uas_id": drone,
                            "t_rx_ms": float(r["time_ms"]),
                            "t_gen_ms": t_gen_ms,
                            "gen_clock": kind,
                            "lat_deg": float(lat),
                            "lon_deg": float(lon),
                            "alt_m_msl": float(pos.get("altitude", np.nan)),
                            "gs_mps": float(spd.get("GS", np.nan)),
                            "track_angle_deg": float(spd.get("track_angle", np.nan)),
                            "vert_speed_mps": float(spd.get("vert_speed", 0.0) or 0.0),
                            "yaw_deg": float(att.get("yaw", np.nan)),
                            "is_probe": is_probe,
                            "source_file": f.name,
                        }
                    )

        if not rows:
            raise DatasetNotAvailable(f"bubbles: no telemetry parsed for {scenario_id}")

        df = pd.DataFrame(rows)

        # One time origin per scenario so every aircraft shares a clock.
        t0_ms = float(df["t_rx_ms"].min())
        self._t0[scenario_id] = t0_ms

        frame = GeodeticFrame.from_points(
            df["lat_deg"].to_numpy(), df["lon_deg"].to_numpy(), df["alt_m_msl"].to_numpy()
        )
        self._frames[scenario_id] = frame

        x, y, z = frame.to_enu(
            df["lat_deg"].to_numpy(), df["lon_deg"].to_numpy(), df["alt_m_msl"].to_numpy()
        )
        vx, vy, vz = track_angle_to_enu_velocity(
            df["gs_mps"].to_numpy(),
            df["track_angle_deg"].to_numpy(),
            df["vert_speed_mps"].to_numpy(),
        )

        out = pd.DataFrame(
            {
                "dataset": self.name,
                "scenario_id": scenario_id,
                "uas_id": df["uas_id"],
                # t_s is the *generation* time where known: this is the layer the
                # aircraft actually was in, and the emulator ages it from here.
                "t_s": np.where(
                    np.isfinite(df["t_gen_ms"]),
                    (df["t_gen_ms"] - t0_ms) / 1000.0,
                    (df["t_rx_ms"] - t0_ms) / 1000.0,
                ),
                "lat_deg": df["lat_deg"],
                "lon_deg": df["lon_deg"],
                "alt_m_msl": df["alt_m_msl"],
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_mps": vx,
                "vy_mps": vy,
                "vz_mps": vz,
                "speed_mps": df["gs_mps"],
                "heading_deg": df["track_angle_deg"] % 360.0,
                "source_type": "ground_truth",
                "quality": np.nan,
            }
        )
        # Extra columns kept alongside the schema for the latency analysis.
        out["t_rx_s"] = (df["t_rx_ms"] - t0_ms) / 1000.0
        out["gen_clock"] = df["gen_clock"]
        out["is_probe"] = df["is_probe"]
        out["latency_s"] = np.where(
            df["gen_clock"].to_numpy() == "epoch",
            (df["t_rx_ms"] - df["t_gen_ms"]) / 1000.0,
            np.nan,
        )

        out = out.dropna(subset=["t_s", "x_m", "y_m"])
        out = out.sort_values(["uas_id", "t_s"])
        # Duplicate generation timestamps break the strictly-increasing invariant
        # the state store relies on; keep the first arrival of each.
        out = out.drop_duplicates(subset=["uas_id", "t_s"], keep="first").reset_index(drop=True)

        self._stats[scenario_id] = {
            "clock_kind": {f"{d}|{k}": v for (d, k), v in clock_kind.items()},
            "n_messages": int(len(out)),
            "n_drones": int(out["uas_id"].nunique()),
            "probe_messages_included": self.include_probe_messages,
        }

        aux = ["t_rx_s", "gen_clock", "latency_s", "is_probe"]
        extra = out[aux]
        result = coerce_trajectory_frame(out)
        result = derive_kinematics(result)
        result[aux] = extra
        self._cache[scenario_id] = result
        return result

    # --------------------------------------------------------------- observed

    def load_observations(self, scenario_id: str) -> pd.DataFrame:
        """UTM surveillance tracks, with the system's own measured AoI.

        Returns one row per (cycle, tracked aircraft) with ``aoi_s = t - t_last``
        exactly as the operational system reported it.
        """
        if scenario_id in self._obs_cache:
            return self._obs_cache[scenario_id]

        # Ground truth first: it fixes the ENU frame and the time origin.
        self.load_scenario(scenario_id)
        frame = self._frames[scenario_id]
        t0_ms = self._t0[scenario_id]

        track_root = self._day_dir(scenario_id) / "utm_tracks"
        if not track_root.is_dir():
            raise DatasetNotAvailable(f"bubbles: missing {track_root}")

        rows: list[dict] = []
        offsets: list[float] = []
        skipped = 0

        for f in sorted(track_root.glob("*.txt")):
            for rec in _iter_json_lines(f):
                msg = rec.get("message")
                if not isinstance(msg, list):
                    skipped += 1
                    continue
                cycle_s = float(rec["time_ms"]) / 1000.0
                for tr in msg:
                    if not isinstance(tr, dict):
                        skipped += 1
                        continue
                    pos = tr.get("pos") or []
                    vel = tr.get("vel") or []
                    if len(pos) < 2:
                        continue
                    t = tr.get("t")
                    t_last = tr.get("t_last")
                    if isinstance(t, int | float):
                        offsets.append(float(t) - cycle_s)
                    rows.append(
                        {
                            "uas_id": (tr.get("tracking_ids") or {}).get("telemetry_id"),
                            "t_cycle_s": cycle_s,
                            "t_track": float(t) if isinstance(t, int | float) else np.nan,
                            "t_last": (
                                float(t_last) if isinstance(t_last, int | float) else np.nan
                            ),
                            "lon_deg": float(pos[0]),
                            "lat_deg": float(pos[1]),
                            "alt_m_msl": float(pos[2]) if len(pos) > 2 else np.nan,
                            "vel0": float(vel[0]) if len(vel) > 0 else np.nan,
                            "vel1": float(vel[1]) if len(vel) > 1 else np.nan,
                            "vel2": float(vel[2]) if len(vel) > 2 else np.nan,
                            "tactical_conflict": bool(
                                (tr.get("tactical_conflict") or {}).get("alert", False)
                            ),
                            "comm_alert": bool(
                                any(
                                    x
                                    for x in (tr.get("communication_alert") or {}).get(
                                        "type", []
                                    )
                                )
                            ),
                            "source_file": f.name,
                        }
                    )

        if not rows:
            raise DatasetNotAvailable(f"bubbles: no utm_tracks parsed for {scenario_id}")

        df = pd.DataFrame(rows)

        # Estimate, rather than assume, the track-clock offset (~+7200 s).
        clock_offset_s = float(np.median(offsets)) if offsets else 0.0
        self._offsets[scenario_id] = clock_offset_s

        df["aoi_s"] = df["t_track"] - df["t_last"]
        df["t_s"] = (df["t_track"] - clock_offset_s) - t0_ms / 1000.0
        df["t_gen_s"] = (df["t_last"] - clock_offset_s) - t0_ms / 1000.0

        x, y, z = frame.to_enu(
            df["lat_deg"].to_numpy(),
            df["lon_deg"].to_numpy(),
            df["alt_m_msl"].fillna(0.0).to_numpy(),
        )
        df["x_m"], df["y_m"], df["z_m"] = x, y, z
        # The track `vel` triple is ENU in m/s, matching the [lon, lat, alt]
        # ordering of `pos`.  Verified against ground truth on three airframes:
        # vel0 correlates with the true East component (r = +0.78 to +0.82) and
        # vel1 with the true North component (r = +0.68 to +0.84).
        df["vx_mps"] = df["vel0"]
        df["vy_mps"] = df["vel1"]
        df["vz_mps"] = df["vel2"]
        df["speed_mps"] = np.hypot(df["vx_mps"], df["vy_mps"])
        df["heading_deg"] = (
            np.degrees(np.arctan2(df["vx_mps"], df["vy_mps"])) % 360.0
        )
        # The UTM also tracked the injected probe messages (same telemetry_id,
        # position ~40 km NW).  Flag them so conflict analysis can exclude them
        # while the AoI statistics, which are genuine system behaviour, stay.
        df["is_probe"] = np.hypot(df["x_m"], df["y_m"]) > _PROBE_DISTANCE_M
        df["dataset"] = self.name
        df["scenario_id"] = scenario_id
        df["source_type"] = "observed"

        df = df.dropna(subset=["uas_id", "t_s"]).sort_values(["uas_id", "t_s"])
        df = df.reset_index(drop=True)

        self._stats.setdefault(scenario_id, {})["observations"] = {
            "n_rows": int(len(df)),
            "n_skipped_records": int(skipped),
            "clock_offset_s": clock_offset_s,
            "n_tactical_alerts": int(df["tactical_conflict"].sum()),
            "n_probe_tracks": int(df["is_probe"].sum()),
        }
        self._obs_cache[scenario_id] = df
        return df

    # ------------------------------------------------------------------ misc

    def frame(self, scenario_id: str) -> GeodeticFrame:
        if scenario_id not in self._frames:
            self.load_scenario(scenario_id)
        return self._frames[scenario_id]

    def provenance(self, scenario_id: str) -> Provenance:
        day = self._day_dir(scenario_id)
        p = Provenance(
            source_dataset="BUBBLES",
            scenario_id=scenario_id,
            coordinate_frame="ENU",
            origin=self.frame(scenario_id).as_dict(),
            notes=[
                "ground truth = onboard telemetry, timestamped at generation "
                "(message.time) where that clock is an epoch clock",
                "observed = utm_tracks; aoi_s = t - t_last as reported by the "
                "operational UTM, not recomputed",
                f"utm_tracks clock offset estimated at "
                f"{self._offsets.get(scenario_id, float('nan')):.1f} s",
            ],
        )
        p.add_source(str(day / "input" / "telemetry"))
        p.add_source(str(day / "utm_tracks"))
        p.extra = self._stats.get(scenario_id, {})
        return p
