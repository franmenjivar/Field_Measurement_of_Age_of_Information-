"""Flight-phase segmentation for the manoeuvre-sensitivity experiment.

Research plan section 17 asks whether operational dynamics change the freshness
requirement.  Answering it needs the reference trajectory labelled by what the
aircraft was doing, so that "1 second of age" can be priced separately in
steady flight and in a turn.

Labels are derived from the reference trajectory itself (turn rate, vertical
speed, along-track acceleration), not from autopilot mode flags, so the same
rule applies to every dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASES = ("cruise", "turn", "climb", "descent", "accel")


@dataclass(frozen=True)
class PhaseThresholds:
    """Cut-offs separating steady flight from manoeuvring.

    Defaults are set for small fixed-wing and multirotor UAS.  They are
    experiment parameters and should be reported with the results.
    """

    turn_rate_deg_s: float = 3.0
    vertical_speed_mps: float = 1.0
    accel_mps2: float = 0.5
    #: smoothing window in seconds, to keep sensor noise out of the labels
    smooth_s: float = 1.0


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def label_phases(
    traj: pd.DataFrame, thresholds: PhaseThresholds | None = None
) -> pd.DataFrame:
    """Add ``phase``, ``turn_rate_deg_s``, ``accel_mps2`` per aircraft.

    Priority when several conditions hold at once: turn > climb/descent >
    acceleration > cruise.  Turning dominates because it is the case that most
    strongly breaks constant-velocity extrapolation.
    """
    th = thresholds or PhaseThresholds()
    out = []

    for _, g in traj.groupby(["scenario_id", "uas_id"], sort=False, observed=True):
        g = g.sort_values("t_s").copy()
        t = g["t_s"].to_numpy(float)
        if len(t) < 5:
            g["phase"] = "cruise"
            g["turn_rate_deg_s"] = np.nan
            g["accel_mps2"] = np.nan
            out.append(g)
            continue

        dt = np.median(np.diff(t))
        n = max(1, int(round(th.smooth_s / dt)) | 1)  # odd window

        hdg = np.radians(g["heading_deg"].to_numpy(float))
        # Unwrap before differentiating so the 360 -> 0 wrap is not a huge rate.
        turn_rate = np.degrees(np.gradient(np.unwrap(hdg), t))
        turn_rate = _smooth(turn_rate, n)

        speed = g["speed_mps"].to_numpy(float)
        accel = _smooth(np.gradient(speed, t), n)
        vz = _smooth(g["vz_mps"].to_numpy(float), n)

        phase = np.full(len(g), "cruise", dtype=object)
        phase[np.abs(accel) > th.accel_mps2] = "accel"
        phase[vz < -th.vertical_speed_mps] = "descent"
        phase[vz > th.vertical_speed_mps] = "climb"
        phase[np.abs(turn_rate) > th.turn_rate_deg_s] = "turn"

        g["phase"] = phase
        g["turn_rate_deg_s"] = turn_rate
        g["accel_mps2"] = accel
        out.append(g)

    return pd.concat(out).sort_index()


def attach_phase(
    errors: pd.DataFrame, labelled: pd.DataFrame, *, time_col: str = "t_s"
) -> pd.DataFrame:
    """Join phase labels onto a freshness-error table by nearest time.

    The error table is evaluated on its own decision grid, so an exact join
    would drop nearly everything; nearest-time is correct here because the
    reference stream is far denser than the decision grid.
    """
    out = []
    for uid, g in errors.groupby("uas_id", sort=False):
        ref = labelled[labelled["uas_id"] == uid].sort_values(time_col)
        if ref.empty:
            g = g.copy()
            g["phase"] = pd.NA
            out.append(g)
            continue
        merged = pd.merge_asof(
            g.sort_values(time_col),
            ref[[time_col, "phase", "turn_rate_deg_s", "accel_mps2"]],
            on=time_col,
            direction="nearest",
        )
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def phase_summary(df: pd.DataFrame, *, err_col: str = "horiz_error_m") -> pd.DataFrame:
    """Error statistics per phase, plus the amplification relative to cruise."""
    g = (
        df.groupby("phase", observed=True)[err_col]
        .agg(n="size", median="median", p95=lambda s: np.percentile(s, 95), max="max")
        .reindex([p for p in PHASES if p in set(df["phase"].dropna())])
        .dropna(how="all")
    )
    if "cruise" in g.index and g.loc["cruise", "median"]:
        g["amplification_vs_cruise"] = g["median"] / g.loc["cruise", "median"]
    return g.reset_index()
