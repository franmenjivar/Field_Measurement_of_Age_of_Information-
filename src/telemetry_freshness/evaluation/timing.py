"""Alert timing: when the conflict was declared versus when it mattered.

Recall alone cannot tell you whether a surveillance system is safe.  These
functions turn a time series of per-pair assessments into the two quantities
that decide whether an alert was useful: how much later than ideal it arrived,
and how much manoeuvring time was left when it did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EncounterTiming:
    """Timing outcome for one aircraft pair over one encounter."""

    uas_a: str
    uas_b: str
    t_ideal_alert_s: float
    t_observed_alert_s: float
    t_los_s: float
    detection_delay_s: float
    alert_lead_time_s: float
    min_separation_truth_m: float
    min_separation_observed_m: float
    truth_conflict: bool
    observed_conflict: bool
    missed: bool
    false_alarm: bool

    def as_dict(self) -> dict:
        return self.__dict__ if not hasattr(self, "__slots__") else {
            f: getattr(self, f) for f in self.__annotations__
        }


def _first_true_time(times: np.ndarray, flags: np.ndarray) -> float:
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    return float(times[idx[0]]) if idx.size else np.nan


def encounter_timing(
    pair_series: pd.DataFrame,
    *,
    time_col: str = "t_s",
    truth_conflict_col: str = "conflict_truth",
    obs_conflict_col: str = "conflict_observed",
    truth_sep_col: str = "separation_truth_m",
    obs_sep_col: str = "separation_observed_m",
    separation_threshold_m: float = 30.0,
) -> EncounterTiming:
    """Reduce one pair's decision-time series to a single timing outcome.

    ``t_los`` is the first moment true separation breaches the threshold; lead
    time is measured against that, not against CPA, because the loss of
    separation is the event the operator must avoid.
    """
    d = pair_series.sort_values(time_col)
    t = d[time_col].to_numpy(dtype=float)

    t_ideal = _first_true_time(t, d[truth_conflict_col].to_numpy())
    t_obs = _first_true_time(t, d[obs_conflict_col].to_numpy())

    sep_truth = d[truth_sep_col].to_numpy(dtype=float)
    t_los = _first_true_time(t, sep_truth < separation_threshold_m)

    truth_conflict = np.isfinite(t_ideal)
    obs_conflict = np.isfinite(t_obs)

    return EncounterTiming(
        uas_a=str(d["uas_a"].iloc[0]),
        uas_b=str(d["uas_b"].iloc[0]),
        t_ideal_alert_s=t_ideal,
        t_observed_alert_s=t_obs,
        t_los_s=t_los,
        detection_delay_s=(t_obs - t_ideal) if (truth_conflict and obs_conflict) else np.nan,
        # Negative lead time means the alert came after separation was already
        # lost; kept signed rather than clipped, since that is the failure case.
        alert_lead_time_s=(t_los - t_obs) if (obs_conflict and np.isfinite(t_los)) else np.nan,
        min_separation_truth_m=float(np.nanmin(sep_truth)) if sep_truth.size else np.nan,
        min_separation_observed_m=(
            float(np.nanmin(d[obs_sep_col].to_numpy(dtype=float)))
            if obs_sep_col in d
            else np.nan
        ),
        truth_conflict=bool(truth_conflict),
        observed_conflict=bool(obs_conflict),
        missed=bool(truth_conflict and not obs_conflict),
        false_alarm=bool(obs_conflict and not truth_conflict),
    )


def timing_table(
    records: pd.DataFrame, *, group_cols: tuple[str, ...] = ("uas_a", "uas_b"), **kwargs
) -> pd.DataFrame:
    """Apply :func:`encounter_timing` to every pair in a record table."""
    rows = []
    for key, g in records.groupby(list(group_cols), sort=False):
        r = encounter_timing(g, **kwargs)
        row = {f: getattr(r, f) for f in EncounterTiming.__dataclass_fields__}
        if isinstance(key, tuple):
            row.update(dict(zip(group_cols, key, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)
