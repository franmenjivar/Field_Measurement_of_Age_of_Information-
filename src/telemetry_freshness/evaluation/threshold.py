"""Deriving the safe Age-of-Information threshold.

The paper's headline quantity (research plan, section 16):

    AoI_max^safe = max{ AoI : all stated performance constraints hold }

Two things matter for defensibility.  First, the constraints are *inputs*, not
findings -- the caller states them and they are carried into the result so a
reader can see what was assumed.  Second, a bin only counts as satisfying a
constraint if it does so with statistical confidence; a bin with eleven samples
and a lucky recall of 1.0 must not set the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import AOI_BIN_EDGES_S, bin_by_aoi, detection_counts
from .statistics import wilson_interval


@dataclass(frozen=True)
class SafetyConstraints:
    """Performance requirements the surveillance system must meet.

    These are experiment assumptions and must be justified in the paper from
    literature or sensitivity analysis -- never presented as regulation.
    """

    min_recall: float = 0.99
    min_lead_time_p95_s: float = 3.0
    max_ttc_error_p95_s: float | None = None
    #: a bin needs at least this many observations to be trusted
    min_samples: int = 50
    #: require the constraint to hold at the lower confidence bound, not the point estimate
    use_confidence_bound: bool = True
    confidence: float = 0.95

    def describe(self) -> dict:
        return {
            "min_recall": self.min_recall,
            "min_lead_time_p95_s": self.min_lead_time_p95_s,
            "max_ttc_error_p95_s": self.max_ttc_error_p95_s,
            "min_samples": self.min_samples,
            "use_confidence_bound": self.use_confidence_bound,
            "confidence": self.confidence,
        }


@dataclass
class ThresholdResult:
    """Outcome of a threshold derivation for one condition."""

    aoi_max_safe_s: float
    limiting_constraint: str | None
    bins: pd.DataFrame
    constraints: dict = field(default_factory=dict)
    condition: dict = field(default_factory=dict)
    #: bins passed over for lack of evidence, not for failing a constraint
    skipped_bins: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return np.isfinite(self.aoi_max_safe_s)


def evaluate_bins(
    df: pd.DataFrame,
    constraints: SafetyConstraints,
    *,
    aoi_column: str = "aoi_s",
    truth_col: str = "conflict_truth",
    obs_col: str = "conflict_observed",
    lead_col: str | None = "alert_lead_time_s",
    ttc_err_col: str | None = "ttc_error_abs_s",
    edges: tuple[float, ...] = AOI_BIN_EDGES_S,
) -> pd.DataFrame:
    """Per-AoI-bin performance plus a pass/fail verdict per constraint."""
    binned = bin_by_aoi(df, aoi_column=aoi_column, edges=edges)
    rows = []

    for interval, g in binned.groupby("aoi_bin", observed=True, sort=True):
        counts = detection_counts(g[truth_col].to_numpy(), g[obs_col].to_numpy())
        n_pos = counts.tp + counts.fn
        recall_lo, recall_hi = (
            wilson_interval(counts.tp, n_pos, constraints.confidence)
            if n_pos
            else (np.nan, np.nan)
        )
        recall_test = recall_lo if constraints.use_confidence_bound else counts.recall

        row = {
            "aoi_bin": str(interval),
            "aoi_left_s": float(interval.left),
            "aoi_right_s": float(interval.right),
            "aoi_mean_s": float(g[aoi_column].mean()),
            "n": int(len(g)),
            "n_positive": int(n_pos),
            "recall": counts.recall,
            "recall_lo": recall_lo,
            "recall_hi": recall_hi,
            "precision": counts.precision,
        }

        ok_samples = len(g) >= constraints.min_samples
        ok_recall = bool(np.isfinite(recall_test) and recall_test >= constraints.min_recall)

        ok_lead, lead_p5 = True, np.nan
        if lead_col and lead_col in g:
            lead = g[lead_col].dropna().to_numpy(dtype=float)
            if lead.size:
                # P95 *lead time* requirement means the 5th percentile of the
                # lead-time distribution: 95% of alerts must be at least this early.
                lead_p5 = float(np.percentile(lead, 5))
                ok_lead = lead_p5 >= constraints.min_lead_time_p95_s
        row["lead_time_p5_s"] = lead_p5

        ok_ttc, ttc_p95 = True, np.nan
        if constraints.max_ttc_error_p95_s is not None and ttc_err_col and ttc_err_col in g:
            e = g[ttc_err_col].dropna().to_numpy(dtype=float)
            if e.size:
                ttc_p95 = float(np.percentile(e, 95))
                ok_ttc = ttc_p95 <= constraints.max_ttc_error_p95_s
        row["ttc_error_p95_s"] = ttc_p95

        row.update(
            {
                "ok_samples": ok_samples,
                "ok_recall": ok_recall,
                "ok_lead_time": ok_lead,
                "ok_ttc_error": ok_ttc,
                "satisfies": bool(ok_samples and ok_recall and ok_lead and ok_ttc),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def derive_safe_aoi(
    df: pd.DataFrame,
    constraints: SafetyConstraints | None = None,
    *,
    condition: dict | None = None,
    **kwargs,
) -> ThresholdResult:
    """Largest AoI bin whose constraints hold, with no failing bin below it.

    The "no failing bin below it" rule matters: performance is not guaranteed
    monotonic in AoI, and a threshold read off an isolated passing bin above a
    failing one would not be operationally meaningful.
    """
    constraints = constraints or SafetyConstraints()
    bins = evaluate_bins(df, constraints, **kwargs)
    if bins.empty:
        return ThresholdResult(
            np.nan, "no_data", bins, constraints.describe(), condition or {}, [])

    bins = bins.sort_values("aoi_left_s").reset_index(drop=True)

    threshold = np.nan
    limiting = None
    skipped: list[str] = []

    for _, r in bins.iterrows():
        if not r["ok_samples"]:
            # Not enough evidence to accept or reject this bin.  Skip it rather
            # than treating it as a wall: a sparse bin is missing information,
            # not a failure, and stopping there would let one thin bin void the
            # whole derivation.  The skipped ranges are reported so the gap in
            # evidence stays visible, and the threshold can only be set by a bin
            # that actually passed.
            skipped.append(str(r["aoi_bin"]))
            continue
        if r["satisfies"]:
            threshold = float(r["aoi_right_s"])
            continue
        if not r["ok_recall"]:
            limiting = "recall"
        elif not r["ok_lead_time"]:
            limiting = "lead_time"
        else:
            limiting = "ttc_error"
        break

    if limiting is None and np.isnan(threshold):
        limiting = "insufficient_samples"

    return ThresholdResult(
        aoi_max_safe_s=threshold,
        limiting_constraint=limiting,
        bins=bins,
        constraints=constraints.describe(),
        condition=condition or {},
        skipped_bins=skipped,
    )


def safe_aoi_by_condition(
    df: pd.DataFrame,
    group_cols: list[str],
    constraints: SafetyConstraints | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Threshold per operating condition -- the data behind Figure 6."""
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        cond = dict(zip(group_cols, key if isinstance(key, tuple) else (key,), strict=True))
        res = derive_safe_aoi(g, constraints, condition=cond, **kwargs)
        rows.append(
            {
                **cond,
                "aoi_max_safe_s": res.aoi_max_safe_s,
                "limiting_constraint": res.limiting_constraint,
                "n_observations": int(len(g)),
                "n_skipped_bins": len(res.skipped_bins),
            }
        )
    return pd.DataFrame(rows)
