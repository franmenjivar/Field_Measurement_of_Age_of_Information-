"""Confidence intervals for the reported metrics.

Rare late-detection events are exactly the ones that matter for safety, so
point estimates are never reported alone (research plan, section 23).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because recall here sits very close
    to 1.0, where the Wald interval famously misbehaves and can exceed 1.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap ``(point, lo, hi)`` for any statistic."""
    a = pd.Series(values).dropna().to_numpy(dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    point = float(statistic(a))
    if a.size < 2:
        return (point, float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.apply_along_axis(statistic, 1, a[idx])
    lo, hi = np.percentile(boots, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return (point, float(lo), float(hi))


def summarise_with_ci(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    *,
    statistic=np.mean,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Grouped point estimate with bootstrap CI plus the usual percentiles."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        vals = g[value_col].dropna().to_numpy(dtype=float)
        point, lo, hi = bootstrap_ci(
            vals, statistic, n_boot=n_boot, confidence=confidence, rng=rng
        )
        cond = dict(zip(group_cols, key if isinstance(key, tuple) else (key,), strict=True))
        rows.append(
            {
                **cond,
                "n": int(vals.size),
                "point": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "p95": float(np.percentile(vals, 95)) if vals.size else np.nan,
                "p99": float(np.percentile(vals, 99)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spearman_predictors(
    df: pd.DataFrame, target: str, predictors: list[str]
) -> pd.DataFrame:
    """Rank correlation of each predictor with the target.

    Used for RQ3: whether freshness requirements track global density, local
    interaction density, or conflict interaction rate.  Rank-based because the
    relationships are expected to be monotonic but not linear.
    """
    rows = []
    for p in predictors:
        d = df[[p, target]].dropna()
        if len(d) < 3:
            rows.append({"predictor": p, "spearman_rho": np.nan, "p_value": np.nan, "n": len(d)})
            continue
        rho, pv = stats.spearmanr(d[p], d[target])
        rows.append(
            {"predictor": p, "spearman_rho": float(rho), "p_value": float(pv), "n": int(len(d))}
        )
    return pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)
