"""Inference that respects the dependence structure of the data.

Two kinds of dependence invalidate naive intervals in this study and both are
severe enough to change reported significance by orders of magnitude:

**Serial dependence.**  Consecutive UTM surveillance cycles are ~1 s apart and
the number of tracked aircraft has a lag-1 autocorrelation of 0.998.  Treating
cycles as independent inflates the effective sample size by 6x to 29x.  Use
:func:`block_bootstrap_spearman`, which resamples contiguous blocks and so
preserves the local correlation structure.

**Clustered outcomes.**  Detection outcomes are recorded per decision epoch, but
epochs within one encounter are strongly dependent -- a pair that is missed at
one epoch is very likely missed at the next.  A Wilson interval over epochs is
therefore far too narrow.  Use :func:`cluster_bootstrap_proportion`, which
resamples whole encounters.

Neither correction is optional: applied to this study's own results, the
uncorrected day-2 load correlation reads p ~ 1e-78 and the corrected one reads
p ~ 1e-05.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval and its effective sample size."""

    point: float
    lo: float
    hi: float
    n: int
    n_effective: float
    p_value: float | None = None
    method: str = ""

    def as_dict(self) -> dict:
        return {
            "point": self.point, "ci_lo": self.lo, "ci_hi": self.hi,
            "n": self.n, "n_effective": self.n_effective,
            "p_value": self.p_value, "method": self.method,
        }

    def __str__(self) -> str:  # pragma: no cover - display helper
        p = "" if self.p_value is None else f", p={self.p_value:.2e}"
        return (f"{self.point:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}] "
                f"(n={self.n:,}, n_eff≈{self.n_effective:,.0f}{p})")


def effective_sample_size(x, y=None) -> float:
    """Sample size adjusted for lag-1 serial correlation.

    Uses the standard AR(1) correction ``n (1-r1 r2) / (1+r1 r2)``.  It is a
    crude approximation -- the block bootstrap is the primary tool -- but it is
    useful for reporting how much independent information a series carries.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if y is None:
        r2 = r1
    else:
        y = np.asarray(y, dtype=float)
        r2 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    if not np.isfinite(r1) or not np.isfinite(r2):
        return float(n)
    prod = np.clip(r1 * r2, -0.999, 0.999)
    return float(max(n * (1 - prod) / (1 + prod), 2.0))


def _moving_blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block bootstrap replicate of length ~n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n] % n


def block_bootstrap_spearman(
    x,
    y,
    *,
    block: int | None = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Spearman correlation with a moving-block bootstrap interval.

    ``block`` defaults to ``n**(1/3)`` rounded up, the usual rule of thumb for
    a moving-block bootstrap; it should exceed the correlation length of the
    series.  The p-value is the two-sided bootstrap probability that the
    correlation has the opposite sign to the point estimate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 10:
        return Estimate(np.nan, np.nan, np.nan, n, float(n), None, "block_bootstrap")

    block = block or max(int(np.ceil(n ** (1 / 3))), 2)
    rng = np.random.default_rng(seed)
    point = float(stats.spearmanr(x, y).statistic)

    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = _moving_blocks(n, block, rng)
        boots[i] = stats.spearmanr(x[idx], y[idx]).statistic
    boots = boots[np.isfinite(boots)]

    a = (1 - confidence) / 2
    lo, hi = np.percentile(boots, [a * 100, (1 - a) * 100])
    # Two-sided bootstrap p: how often the replicate crosses zero.
    frac_opposite = float(np.mean(boots <= 0) if point > 0 else np.mean(boots >= 0))
    p = min(1.0, 2 * max(frac_opposite, 1 / (len(boots) + 1)))

    return Estimate(point, float(lo), float(hi), n,
                    effective_sample_size(x, y), p,
                    f"moving-block bootstrap (block={block}, B={n_boot})")


def cluster_bootstrap_proportion(
    successes,
    cluster_ids,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Proportion with a cluster bootstrap interval, resampling whole clusters.

    For detection metrics the cluster is the encounter (or the pair-day), not
    the decision epoch: outcomes within an encounter are strongly dependent, so
    an epoch-level Wilson interval understates the uncertainty substantially.
    """
    s = np.asarray(successes, dtype=bool)
    c = np.asarray(cluster_ids)
    ok = ~pd.isna(c)
    s, c = s[ok], c[ok]
    if s.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0, 0.0, None, "cluster_bootstrap")

    groups = pd.Series(s).groupby(pd.Series(c))
    per = [(g.sum(), g.size) for _, g in groups]
    hits = np.array([p[0] for p in per], dtype=float)
    sizes = np.array([p[1] for p in per], dtype=float)
    k = len(per)

    point = float(hits.sum() / sizes.sum())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        tot = sizes[idx].sum()
        boots[i] = hits[idx].sum() / tot if tot else np.nan
    boots = boots[np.isfinite(boots)]

    a = (1 - confidence) / 2
    lo, hi = np.percentile(boots, [a * 100, (1 - a) * 100])
    return Estimate(point, float(lo), float(hi), int(sizes.sum()), float(k), None,
                    f"cluster bootstrap ({k} clusters, B={n_boot})")


def cluster_bootstrap_statistic(
    values,
    cluster_ids,
    statistic=np.median,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Any statistic of a value column, with whole clusters resampled."""
    v = pd.Series(np.asarray(values, dtype=float))
    c = pd.Series(np.asarray(cluster_ids))
    keep = v.notna() & c.notna()
    v, c = v[keep].to_numpy(), c[keep].to_numpy()
    if v.size == 0:
        return Estimate(np.nan, np.nan, np.nan, 0, 0.0, None, "cluster_bootstrap")

    by = pd.Series(v).groupby(pd.Series(c))
    chunks = [g.to_numpy() for _, g in by]
    k = len(chunks)
    point = float(statistic(v))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boots[i] = statistic(np.concatenate([chunks[j] for j in idx]))
    boots = boots[np.isfinite(boots)]

    a = (1 - confidence) / 2
    lo, hi = np.percentile(boots, [a * 100, (1 - a) * 100])
    return Estimate(point, float(lo), float(hi), int(v.size), float(k), None,
                    f"cluster bootstrap ({k} clusters, B={n_boot})")


def cluster_bootstrap_difference(
    successes_a,
    clusters_a,
    successes_b,
    clusters_b,
    *,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Difference of two proportions that share their clusters.

    Comparing two independently-bootstrapped intervals and checking whether they
    overlap is both conservative and needlessly noisy.  The two groups here are
    two AoI regimes of the *same* aircraft pairs, so the clusters are resampled
    **jointly** and the difference is formed inside each replicate.  ``p_value``
    is the one-sided bootstrap probability that the difference is <= 0.
    """
    a = pd.DataFrame({"y": np.asarray(successes_a, dtype=bool),
                      "c": np.asarray(clusters_a)})
    b = pd.DataFrame({"y": np.asarray(successes_b, dtype=bool),
                      "c": np.asarray(clusters_b)})
    clusters = sorted(set(a["c"]) | set(b["c"]))
    ga = {k: v["y"].to_numpy() for k, v in a.groupby("c")}
    gb = {k: v["y"].to_numpy() for k, v in b.groupby("c")}

    point = float(b["y"].mean() - a["y"].mean())
    rng = np.random.default_rng(seed)
    k = len(clusters)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, size=k)
        A = np.concatenate([ga.get(clusters[i], np.array([], bool)) for i in idx])
        B = np.concatenate([gb.get(clusters[i], np.array([], bool)) for i in idx])
        if A.size and B.size:
            boots.append(B.mean() - A.mean())
    boots = np.asarray(boots)
    if boots.size == 0:
        return Estimate(point, np.nan, np.nan, len(a) + len(b), float(k), None,
                        "paired cluster bootstrap")

    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    p = float((boots <= 0).mean()) if point > 0 else float((boots >= 0).mean())
    return Estimate(point, float(lo), float(hi), len(a) + len(b), float(k), p,
                    f"paired cluster bootstrap ({k} clusters, B={boots.size})")
