"""Does surveillance freshness degrade under traffic load, in a real system?

The research plan is explicit (section 24) that one must *not* assume high
density causes stale data.  BUBBLES lets the question be answered rather than
assumed: it records both the number of aircraft the UTM was tracking at each
surveillance cycle and the age of each aircraft's state at that cycle.

The analysis deliberately separates the centre of the age distribution from its
tail, because they behave differently and only the tail turns out to respond to
load.  Time-in-campaign is controlled for, since traffic builds up over a
campaign and a raw correlation would partly measure that instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class LoadCorrelation:
    """Association between concurrent traffic and one freshness statistic."""

    statistic: str
    rho_raw: float
    p_raw: float
    rho_partial: float
    p_partial: float
    n_cycles: int

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "rho_raw": self.rho_raw,
            "p_raw": self.p_raw,
            "rho_partial_time_controlled": self.rho_partial,
            "p_partial": self.p_partial,
            "n_cycles": self.n_cycles,
        }


def cycle_load_table(
    observations: pd.DataFrame,
    *,
    min_tracked: int = 2,
    exclude_probe: bool = True,
    round_to_s: float = 0.1,
) -> pd.DataFrame:
    """One row per UTM surveillance cycle: aircraft tracked and age statistics.

    Cycles tracking fewer than ``min_tracked`` aircraft are dropped: a
    single-track cycle has no meaningful percentile, and in this dataset those
    cycles are idle-system artefacts carrying multi-minute ages.
    """
    o = observations
    if exclude_probe and "is_probe" in o.columns:
        o = o[~o["is_probe"]]
    o = o.dropna(subset=["aoi_s"]).copy()
    o["cycle"] = (o["t_cycle_s"] / round_to_s).round() * round_to_s

    per = o.groupby("cycle").agg(
        n_tracked=("uas_id", "nunique"),
        aoi_median_s=("aoi_s", "median"),
        aoi_p95_s=("aoi_s", lambda s: float(np.percentile(s, 95))),
        aoi_max_s=("aoi_s", "max"),
        n_observations=("aoi_s", "size"),
        t_s=("t_s", "median"),
    )
    return per[per["n_tracked"] >= min_tracked].reset_index()


def _partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman association of x and y with z partialled out.

    Implemented on ranks: rank-transform all three, linearly remove z from each
    of x and y, then correlate the residuals.  Rank-based because the age
    distribution is heavy-tailed and a Pearson correlation would be dominated by
    a handful of extreme cycles.
    """
    rx, ry, rz = (stats.rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def load_vs_freshness(
    per_cycle: pd.DataFrame,
    *,
    statistics: tuple[str, ...] = ("aoi_median_s", "aoi_p95_s"),
    control_col: str = "t_s",
    load_col: str = "n_tracked",
) -> pd.DataFrame:
    """Correlate traffic load with each freshness statistic, raw and controlled."""
    rows = []
    for stat in statistics:
        d = per_cycle[[load_col, stat, control_col]].dropna()
        if len(d) < 10 or d[load_col].nunique() < 3:
            continue
        rho, p = stats.spearmanr(d[load_col], d[stat])
        prho, pp = _partial_spearman(d[load_col], d[stat], d[control_col])
        rows.append(
            LoadCorrelation(stat, float(rho), float(p), prho, pp, int(len(d))).as_dict()
        )
    return pd.DataFrame(rows)


def load_profile(
    per_cycle: pd.DataFrame, *, load_col: str = "n_tracked", min_cycles: int = 20
) -> pd.DataFrame:
    """Freshness statistics as a function of how many aircraft were tracked."""
    g = per_cycle.groupby(load_col).agg(
        cycles=("aoi_median_s", "size"),
        aoi_median_s=("aoi_median_s", "median"),
        aoi_p95_s=("aoi_p95_s", "median"),
        aoi_p95_hi=("aoi_p95_s", lambda s: float(np.percentile(s, 75))),
        aoi_p95_lo=("aoi_p95_s", lambda s: float(np.percentile(s, 25))),
    )
    return g[g["cycles"] >= min_cycles].reset_index()


def within_block_robustness(
    per_cycle: pd.DataFrame,
    *,
    stat: str = "aoi_p95_s",
    load_col: str = "n_tracked",
    control_col: str = "t_s",
    n_blocks: int = 4,
) -> pd.DataFrame:
    """Repeat the correlation inside equal-time blocks.

    A relationship that only exists across the whole campaign but not inside any
    block is a time trend wearing a traffic costume.
    """
    d = per_cycle.copy()
    d["block"] = pd.qcut(d[control_col], n_blocks, labels=range(1, n_blocks + 1))
    rows = []
    for b, g in d.groupby("block", observed=True):
        if len(g) < 30 or g[load_col].nunique() < 3:
            continue
        rho, p = stats.spearmanr(g[load_col], g[stat])
        rows.append(
            {
                "block": int(b),
                "n_cycles": int(len(g)),
                "load_min": int(g[load_col].min()),
                "load_max": int(g[load_col].max()),
                "rho": float(rho),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)
