"""Traffic density measures.

The research plan (section 9) is explicit that aircraft count alone is a poor
descriptor: a hundred aircraft flying parallel are less conflict-intensive than
twenty-five crossing.  Three measures are computed so RQ3 can ask which one
actually predicts the freshness requirement.

``global_density``      aircraft per km^2 over the operating area
``local_density``       neighbours within radius r of each aircraft
``interaction_rate``    converging pairs per minute -- the encounter-based view
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

#: Neighbourhood radii from the research plan, section 9.2.
LOCAL_RADII_M: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0)


def _snapshot(traj: pd.DataFrame, t_s: float, tolerance_s: float) -> pd.DataFrame:
    """Nearest sample per aircraft to ``t_s``, within ``tolerance_s``."""
    d = traj.loc[(traj["t_s"] - t_s).abs() <= tolerance_s]
    if d.empty:
        return d
    idx = (d["t_s"] - t_s).abs().groupby(d["uas_id"]).idxmin()
    return d.loc[idx]


def global_density(
    traj: pd.DataFrame, t_s: float, *, area_km2: float | None = None,
    tolerance_s: float = 1.0,
) -> dict:
    """Aircraft per km^2 at one instant.

    When ``area_km2`` is not supplied the area is taken as the convex extent of
    the airborne aircraft, which is the operationally meaningful denominator --
    a fixed nominal area would make density an artefact of the chosen box.
    """
    s = _snapshot(traj, t_s, tolerance_s)
    n = int(s["uas_id"].nunique())
    if n == 0:
        return {"t_s": t_s, "n_aircraft": 0, "area_km2": np.nan, "density_per_km2": np.nan}
    if area_km2 is None:
        dx = (s["x_m"].max() - s["x_m"].min()) / 1000.0
        dy = (s["y_m"].max() - s["y_m"].min()) / 1000.0
        area_km2 = max(dx * dy, 1e-6)
    return {
        "t_s": t_s,
        "n_aircraft": n,
        "area_km2": float(area_km2),
        "density_per_km2": float(n / area_km2),
    }


def local_density(
    traj: pd.DataFrame, t_s: float, *, radii_m=LOCAL_RADII_M, tolerance_s: float = 1.0
) -> pd.DataFrame:
    """Neighbour count within each radius, per aircraft, at one instant."""
    s = _snapshot(traj, t_s, tolerance_s)
    if len(s) < 2:
        return pd.DataFrame()
    pts = s[["x_m", "y_m"]].to_numpy(float)
    tree = cKDTree(pts)
    out = {"t_s": t_s, "uas_id": s["uas_id"].to_numpy()}
    for r in radii_m:
        # Subtract 1: query_ball_point counts the aircraft itself.
        out[f"n_within_{int(r)}m"] = np.array(
            [len(v) - 1 for v in tree.query_ball_point(pts, r)]
        )
    return pd.DataFrame(out)


def interaction_rate(
    records: pd.DataFrame,
    *,
    window_s: float = 60.0,
    converging_only: bool = True,
    conflict_col: str = "conflict_truth",
) -> pd.DataFrame:
    """Potential conflicts and converging pairs per minute.

    Computed from ground-truth assessments, so it describes the encounter
    environment itself rather than what the surveillance system managed to see.
    """
    if records.empty:
        return pd.DataFrame()
    d = records.copy()
    d["window"] = (d["t_s"] // window_s).astype(int)
    rows = []
    for w, g in d.groupby("window", sort=True):
        conv = g["closing_speed_truth_mps"] > 0 if converging_only else slice(None)
        rows.append(
            {
                "window": int(w),
                "t_start_s": float(w * window_s),
                "n_pair_observations": int(len(g)),
                "n_conflict_observations": int(g[conflict_col].sum()),
                "n_converging_pairs": int(
                    g.loc[conv, ["uas_a", "uas_b"]].drop_duplicates().shape[0]
                ),
                "n_conflict_pairs": int(
                    g.loc[g[conflict_col], ["uas_a", "uas_b"]].drop_duplicates().shape[0]
                ),
            }
        )
    out = pd.DataFrame(rows)
    scale = 60.0 / window_s
    out["conflict_pairs_per_min"] = out["n_conflict_pairs"] * scale
    out["converging_pairs_per_min"] = out["n_converging_pairs"] * scale
    return out


def density_timeline(
    traj: pd.DataFrame, *, step_s: float = 10.0, radii_m=LOCAL_RADII_M,
    area_km2: float | None = None,
) -> pd.DataFrame:
    """Global and local density sampled across a scenario."""
    t0, t1 = float(traj["t_s"].min()), float(traj["t_s"].max())
    rows = []
    for t in np.arange(t0, t1, step_s):
        g = global_density(traj, float(t), area_km2=area_km2)
        loc = local_density(traj, float(t), radii_m=radii_m)
        for r in radii_m:
            col = f"n_within_{int(r)}m"
            g[f"local_mean_{int(r)}m"] = float(loc[col].mean()) if not loc.empty else np.nan
            g[f"local_max_{int(r)}m"] = float(loc[col].max()) if not loc.empty else np.nan
        rows.append(g)
    return pd.DataFrame(rows)


def airborne_only(traj: pd.DataFrame, *, min_speed_mps: float = 1.0) -> pd.DataFrame:
    """Drop samples where the aircraft is effectively stationary.

    Campaign recordings contain long periods on the ground; counting those as
    traffic would deflate every density figure.
    """
    return traj[traj["speed_mps"].fillna(0.0) >= min_speed_mps].copy()
