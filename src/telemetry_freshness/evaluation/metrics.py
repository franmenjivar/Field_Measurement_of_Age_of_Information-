"""Comparing the observed detector against the ground-truth detector.

Everything here answers one question: what did information age cost us?
Detection quality (recall and friends), estimation error (signed TTC error) and
timing (detection delay, alert lead time) are reported separately because a
system can be perfect on one and dangerous on another -- detecting every
conflict too late to manoeuvre is not a success (research plan, section 15.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionCounts:
    """Confusion counts over (pair, decision time) observations."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return float(self.tp / d) if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp / d) if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return float(2 * p * r / (p + r)) if p + r > 0 else float("nan")

    @property
    def false_negative_rate(self) -> float:
        d = self.tp + self.fn
        return float(self.fn / d) if d else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn, "n": self.n,
            "recall": self.recall, "precision": self.precision, "f1": self.f1,
            "false_negative_rate": self.false_negative_rate,
        }


def detection_counts(truth: np.ndarray, observed: np.ndarray) -> DetectionCounts:
    """Confusion counts for boolean conflict flags."""
    t = np.asarray(truth, dtype=bool)
    o = np.asarray(observed, dtype=bool)
    if t.shape != o.shape:
        raise ValueError("truth and observed must have the same shape")
    return DetectionCounts(
        tp=int(np.sum(t & o)),
        fp=int(np.sum(~t & o)),
        tn=int(np.sum(~t & ~o)),
        fn=int(np.sum(t & ~o)),
    )


def ttc_error(truth_ttc_s: np.ndarray, observed_ttc_s: np.ndarray) -> pd.DataFrame:
    """Signed and absolute TTC error where both estimates are finite.

    Sign convention: ``error = truth - observed``.

    * **negative** -- observed TTC is *larger* than reality, so the system
      believes it has more time to react than it does.  This is the dangerous
      direction (research plan, section 8: truth 5.2 s vs observed 8.7 s gives
      -3.5 s, "the system believes it has 3.5 seconds more than it has").
    * **positive** -- observed TTC is smaller than reality: a conservative,
      early alert.

    The dangerous direction is flagged separately from the magnitude, because
    the two have opposite operational meaning and averaging them hides it.
    """
    t = np.asarray(truth_ttc_s, dtype=float)
    o = np.asarray(observed_ttc_s, dtype=float)
    both = np.isfinite(t) & np.isfinite(o)
    signed = np.where(both, t - o, np.nan)
    return pd.DataFrame(
        {
            "ttc_truth_s": t,
            "ttc_observed_s": o,
            "ttc_error_signed_s": signed,
            "ttc_error_abs_s": np.abs(signed),
            # observed > truth  <=>  signed < 0  <=>  TTC overestimated
            "ttc_overestimated": np.where(both, signed < 0, False),
            "comparable": both,
        }
    )


def summarise_errors(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Mean/median/P95/P99/max of an error column, ignoring NaN."""
    a = pd.Series(df[column]).dropna().to_numpy(dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


#: Bin edges used throughout the study (research plan, Phase 7).
AOI_BIN_EDGES_S: tuple[float, ...] = (
    0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf,
)


def _bin_midpoint(interval) -> float:
    """Plottable x for an AoI bin; the open top bin is placed at its left edge.

    Written as a function rather than a nested conditional: an untracked
    observation bins to NaN, and the isinstance guard has to run first.
    """
    if not isinstance(interval, pd.Interval):
        return np.nan
    if not np.isfinite(interval.right):
        return float(interval.left)
    return float(0.5 * (interval.left + interval.right))


def bin_by_aoi(
    df: pd.DataFrame, aoi_column: str = "aoi_s", edges: tuple[float, ...] = AOI_BIN_EDGES_S
) -> pd.DataFrame:
    """Add ``aoi_bin`` (interval) and ``aoi_bin_mid`` (plottable) columns."""
    out = df.copy()
    cats = pd.cut(out[aoi_column], bins=list(edges), right=False)
    out["aoi_bin"] = cats
    out["aoi_bin_mid"] = [_bin_midpoint(c) for c in cats]
    return out


def recall_by_aoi(
    df: pd.DataFrame,
    *,
    truth_col: str = "conflict_truth",
    obs_col: str = "conflict_observed",
    aoi_column: str = "aoi_s",
    edges: tuple[float, ...] = AOI_BIN_EDGES_S,
) -> pd.DataFrame:
    """Detection performance per AoI bin -- the shape behind Figure 2."""
    binned = bin_by_aoi(df, aoi_column=aoi_column, edges=edges)
    rows = []
    for interval, g in binned.groupby("aoi_bin", observed=True, sort=True):
        c = detection_counts(g[truth_col].to_numpy(), g[obs_col].to_numpy())
        rows.append(
            {
                "aoi_bin": str(interval),
                "aoi_bin_mid": float(g["aoi_bin_mid"].iloc[0]),
                "aoi_mean_s": float(g[aoi_column].mean()),
                # Recall is only as trustworthy as the number of true conflicts
                # in the bin, so carry that count alongside it.
                "n_positive": int(c.tp + c.fn),
                **c.as_dict(),
            }
        )
    return pd.DataFrame(rows)
