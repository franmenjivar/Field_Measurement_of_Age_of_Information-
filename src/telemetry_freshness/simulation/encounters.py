"""Controlled two-aircraft encounters.

Two constructors, and the difference between them must be stated in the paper:

:func:`synthetic_encounter`
    idealised constant-velocity geometry.  Used to validate the mechanism
    analytically, where the right answer is known in closed form.

:func:`encounter_from_segments`
    real recorded trajectory segments translated and rotated into a chosen
    geometry.  The *motion* is real (real turns, real speed variation, real
    autopilot behaviour); the *encounter geometry is experimentally
    constructed*.  Dataset-strategy section 18 is explicit that this must be
    declared and never presented as a naturally occurring conflict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.schema import coerce_trajectory_frame, derive_kinematics

ENCOUNTER_TYPES = ("head_on", "crossing", "overtaking", "converging", "parallel")

#: Relative bearing between the two tracks, in degrees.
_GEOMETRY_ANGLE_DEG = {
    "head_on": 180.0,
    "crossing": 90.0,
    "overtaking": 0.0,
    "converging": 45.0,
    "parallel": 0.0,
}


@dataclass(frozen=True)
class EncounterSpec:
    """Parameters of a constructed encounter."""

    encounter_type: str
    speed_a_mps: float = 15.0
    speed_b_mps: float = 15.0
    #: horizontal miss distance at the designed CPA
    miss_distance_m: float = 0.0
    #: vertical offset held throughout
    altitude_offset_m: float = 0.0
    #: seconds before the designed CPA that the scenario starts
    lead_in_s: float = 60.0
    duration_s: float = 120.0
    sample_rate_hz: float = 20.0

    def __post_init__(self) -> None:
        if self.encounter_type not in ENCOUNTER_TYPES:
            raise ValueError(
                f"encounter_type must be one of {ENCOUNTER_TYPES}, got {self.encounter_type!r}"
            )


def synthetic_encounter(spec: EncounterSpec, scenario_id: str | None = None) -> pd.DataFrame:
    """Two constant-velocity aircraft meeting at a designed CPA at ``t = lead_in_s``.

    ``parallel`` is deliberately non-conflicting: both aircraft fly the same
    heading offset by ``miss_distance_m``, giving a control case with traffic
    present but no encounter.
    """
    sid = scenario_id or f"SYN-{spec.encounter_type.upper()}"
    t = np.arange(0.0, spec.duration_s, 1.0 / spec.sample_rate_hz)
    tau = t - spec.lead_in_s  # signed time from CPA

    ang = np.radians(_GEOMETRY_ANGLE_DEG[spec.encounter_type])

    # A flies East through the origin; B's heading is rotated by the geometry.
    vax, vay = spec.speed_a_mps, 0.0
    vbx, vby = spec.speed_b_mps * np.cos(ang), spec.speed_b_mps * np.sin(ang)

    ax, ay = vax * tau, np.full_like(t, 0.0)

    if spec.encounter_type == "overtaking":
        # Same track: B ahead and slower, so A closes from behind.
        bx = vbx * tau + spec.lead_in_s * (spec.speed_a_mps - spec.speed_b_mps)
        by = np.full_like(t, spec.miss_distance_m)
    elif spec.encounter_type == "parallel":
        bx, by = vbx * tau, np.full_like(t, max(spec.miss_distance_m, 200.0))
    else:
        # Offset perpendicular to B's track so the miss distance is exact.
        bx = vbx * tau - spec.miss_distance_m * np.sin(ang)
        by = vby * tau + spec.miss_distance_m * np.cos(ang)

    frames = []
    for uid, x, y, vx, vy, z in (
        ("UAS-A", ax, ay, np.full_like(t, vax), np.full_like(t, vay), np.zeros_like(t)),
        (
            "UAS-B",
            bx,
            by,
            np.full_like(t, vbx),
            np.full_like(t, vby),
            np.full_like(t, spec.altitude_offset_m),
        ),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "dataset": "sim",
                    "scenario_id": sid,
                    "uas_id": uid,
                    "t_s": t,
                    "x_m": x,
                    "y_m": y,
                    "z_m": z + 100.0,
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "vz_mps": np.zeros_like(t),
                    "source_type": "ground_truth",
                }
            )
        )

    return derive_kinematics(coerce_trajectory_frame(pd.concat(frames, ignore_index=True)))


def _rigid_transform(x, y, angle_deg: float, dx: float, dy: float):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return c * x - s * y + dx, s * x + c * y + dy


def encounter_from_segments(
    segment_a: pd.DataFrame,
    segment_b: pd.DataFrame,
    spec: EncounterSpec,
    scenario_id: str | None = None,
) -> pd.DataFrame:
    """Place two real trajectory segments into a constructed encounter geometry.

    Both segments are re-referenced to start at ``t = 0`` and at the origin,
    then B is rotated by the geometry angle and translated so that the two
    tracks pass within ``miss_distance_m`` at the point of closest approach.
    Speeds, accelerations and turns are whatever the aircraft actually flew.
    """
    sid = scenario_id or f"REAL-{spec.encounter_type.upper()}"

    def _prep(seg: pd.DataFrame, uid: str) -> pd.DataFrame:
        s = seg.sort_values("t_s").reset_index(drop=True).copy()
        s["t_s"] = s["t_s"] - s["t_s"].iloc[0]
        s["x_m"] = s["x_m"] - s["x_m"].iloc[0]
        s["y_m"] = s["y_m"] - s["y_m"].iloc[0]
        s["uas_id"] = uid
        s["scenario_id"] = sid
        return s

    a = _prep(segment_a, "UAS-A")
    b = _prep(segment_b, "UAS-B")

    ang = _GEOMETRY_ANGLE_DEG[spec.encounter_type]
    bx, by = _rigid_transform(b["x_m"].to_numpy(), b["y_m"].to_numpy(), ang, 0.0, 0.0)
    bvx, bvy = _rigid_transform(b["vx_mps"].to_numpy(), b["vy_mps"].to_numpy(), ang, 0.0, 0.0)

    # Align on a shared time grid, then translate B so the closest approach
    # between the two real tracks equals the requested miss distance.
    t_common = np.arange(
        0.0,
        min(a["t_s"].max(), b["t_s"].max()),
        1.0 / spec.sample_rate_hz,
    )
    ax_i = np.interp(t_common, a["t_s"], a["x_m"])
    ay_i = np.interp(t_common, a["t_s"], a["y_m"])
    bx_i = np.interp(t_common, b["t_s"], bx)
    by_i = np.interp(t_common, b["t_s"], by)

    d = np.hypot(ax_i - bx_i, ay_i - by_i)
    k = int(np.argmin(d))
    dx = ax_i[k] - bx_i[k]
    dy = ay_i[k] - by_i[k]
    if d[k] > 1e-9:
        ux, uy = (bx_i[k] - ax_i[k]) / d[k], (by_i[k] - ay_i[k]) / d[k]
    else:
        ux, uy = 1.0, 0.0
    dx += spec.miss_distance_m * ux
    dy += spec.miss_distance_m * uy

    rows = []
    for uid, x, y, vx, vy, z0 in (
        (
            "UAS-A",
            ax_i,
            ay_i,
            np.interp(t_common, a["t_s"], a["vx_mps"]),
            np.interp(t_common, a["t_s"], a["vy_mps"]),
            np.interp(t_common, a["t_s"], a["z_m"]),
        ),
        (
            "UAS-B",
            bx_i + dx,
            by_i + dy,
            np.interp(t_common, b["t_s"], bvx),
            np.interp(t_common, b["t_s"], bvy),
            np.interp(t_common, b["t_s"], b["z_m"]) + spec.altitude_offset_m,
        ),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "constructed",
                    "scenario_id": sid,
                    "uas_id": uid,
                    "t_s": t_common,
                    "x_m": x,
                    "y_m": y,
                    "z_m": z0,
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "vz_mps": 0.0,
                    "source_type": "ground_truth",
                }
            )
        )

    out = derive_kinematics(coerce_trajectory_frame(pd.concat(rows, ignore_index=True)))
    out.attrs["construction"] = (
        "real trajectory segments, experimentally constructed encounter geometry"
    )
    return out
