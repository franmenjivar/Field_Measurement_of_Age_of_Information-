"""Vectorised multi-aircraft runner for the density experiments.

:func:`telemetry_freshness.experiments.runner.run_scenario` walks every pair in
Python, which is fine for two-aircraft encounters but quadratic and far too slow
at 100 aircraft (4 950 pairs per decision step).  This module produces the same
records with the same semantics, but:

* neighbours are screened with a KD-tree, so only pairs that could possibly
  conflict inside the horizon are evaluated;
* CPA is computed batched over those pairs.

The isolation guarantee is unchanged: the observed side reads only the
surveillance store, and truth and observation go through the same geometry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..conflict.cpa import cpa_vectorised
from ..core.rng import make_rng
from ..surveillance.state_store import SurveillanceStateStore
from ..telemetry.generator import TrajectorySampler, generate_stream
from .runner import ScenarioConfig


def _states_matrix(ids: list[str], lookup: dict[str, tuple]) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([lookup[u][0] for u in ids], dtype=float)
    v = np.array([lookup[u][1] for u in ids], dtype=float)
    return p, v


def _assess(
    ids: list[str],
    pos: np.ndarray,
    vel: np.ndarray,
    pairs: np.ndarray,
    criteria,
) -> dict[tuple[str, str], tuple]:
    """Batched CPA + conflict verdict for the given index pairs."""
    if pairs.size == 0:
        return {}
    ia, ib = pairs[:, 0], pairs[:, 1]
    t_cpa, d_h, d_v = cpa_vectorised(
        pos[ia], vel[ia], pos[ib], vel[ib], horizon_s=criteria.horizon_s
    )
    r = pos[ib] - pos[ia]
    rel_v = vel[ib] - vel[ia]
    rng_now = np.linalg.norm(r, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = np.where(rng_now > 1e-9, -np.einsum("ij,ij->i", r, rel_v) / rng_now, 0.0)

    converging = closing > 0
    conflict = (
        converging
        & (d_h < criteria.horizontal_separation_m)
        & (d_v < criteria.vertical_separation_m)
    )
    ttc = np.where(conflict, t_cpa, np.inf)

    out = {}
    for k in range(pairs.shape[0]):
        a, b = ids[ia[k]], ids[ib[k]]
        key = (a, b) if a < b else (b, a)
        out[key] = (
            bool(conflict[k]), float(ttc[k]), float(d_h[k]), float(d_v[k]),
            float(rng_now[k]), float(closing[k]),
        )
    return out


def run_scenario_fast(
    trajectories: pd.DataFrame,
    config: ScenarioConfig,
    *,
    screen_radius_m: float | None = None,
    record: str = "all",
) -> pd.DataFrame:
    """Same output contract as ``run_scenario``, built for many aircraft.

    Parameters
    ----------
    screen_radius_m
        Pairs beyond this range are not evaluated.  Defaults to a horizon-based
        bound: two aircraft further apart than ``horizon * 2 * v_max`` cannot
        reach separation minima within the horizon, so skipping them changes no
        verdict, only the runtime.
    record
        ``"all"`` emits one row per evaluated pair.  ``"relevant"`` emits only
        pairs that either layer called a conflict, and counts the rest into
        ``df.attrs["n_true_negative"]``.  Every metric this study reports --
        recall, precision, F1, TTC error, timing -- depends only on the
        conflict-relevant rows, so ``"relevant"`` is lossless for them while
        making 100-aircraft scenarios tractable.  Use ``"all"`` when the
        true-negative rows are themselves the subject.
    """
    if record not in {"all", "relevant"}:
        raise ValueError("record must be 'all' or 'relevant'")
    samplers = {
        uid: TrajectorySampler(g)
        for uid, g in trajectories.groupby("uas_id", sort=True)
        if len(g) >= 2
    }
    if len(samplers) < 2:
        return pd.DataFrame()

    t0 = config.t_start if config.t_start is not None else max(s.t_start for s in samplers.values())
    t1 = config.t_end if config.t_end is not None else min(s.t_end for s in samplers.values())
    if t1 <= t0:
        return pd.DataFrame()

    if screen_radius_m is None:
        v_max = float(np.nanmax(trajectories["speed_mps"].to_numpy(dtype=float)))
        screen_radius_m = (
            2.0 * v_max * config.criteria.horizon_s + config.criteria.horizontal_separation_m
        )

    store = SurveillanceStateStore(keep_newest_generation=config.keep_newest_generation)
    for uid, s in samplers.items():
        rng = make_rng(config.seed, f"{config.scenario_id}|{uid}")
        store.ingest(generate_stream(s, config.telemetry, rng, t_start=t0, t_end=t1))

    ids = sorted(samplers)
    crit = config.criteria
    rows = []
    n_true_negative = 0
    n_evaluated = 0

    for t in np.arange(t0, t1, 1.0 / config.decision_rate_hz):
        t = float(t)

        truth = {}
        for uid in ids:
            st = samplers[uid].at(t)
            truth[uid] = (st.position, st.velocity)

        obs, ages = {}, {}
        for uid in ids:
            msg = store.latest(uid, t)
            if msg is None:
                continue
            age = t - msg.t_gen_s
            ages[uid] = age
            st = msg.state.propagate(age) if config.extrapolate else msg.state
            obs[uid] = (st.position, st.velocity)

        tp, tv = _states_matrix(ids, truth)
        # Screen on truth positions: this defines the candidate set, and using
        # one consistent candidate set for both layers keeps the comparison fair.
        tree = cKDTree(tp[:, :2])
        pairs = np.array(sorted(tree.query_pairs(screen_radius_m)), dtype=int)
        if pairs.size == 0:
            continue

        truth_res = _assess(ids, tp, tv, pairs, crit)

        obs_ids = [u for u in ids if u in obs]
        obs_index = {u: i for i, u in enumerate(obs_ids)}
        if len(obs_ids) >= 2:
            op, ov = _states_matrix(obs_ids, obs)
            obs_pairs = np.array(
                [
                    [obs_index[ids[a]], obs_index[ids[b]]]
                    for a, b in pairs
                    if ids[a] in obs_index and ids[b] in obs_index
                ],
                dtype=int,
            )
            obs_res = _assess(obs_ids, op, ov, obs_pairs, crit) if obs_pairs.size else {}
        else:
            obs_res = {}

        for (a, b), (tc, tttc, td_h, td_v, trng, tclose) in truth_res.items():
            o = obs_res.get((a, b))
            age_a, age_b = ages.get(a), ages.get(b)
            n_evaluated += 1
            oc = bool(o[0]) if o else False
            if record == "relevant" and not (tc or oc):
                n_true_negative += 1
                continue
            rows.append(
                {
                    "scenario_id": config.scenario_id,
                    "t_s": t,
                    "uas_a": a,
                    "uas_b": b,
                    "aoi_a_s": np.nan if age_a is None else age_a,
                    "aoi_b_s": np.nan if age_b is None else age_b,
                    "aoi_s": np.nan if (age_a is None or age_b is None) else max(age_a, age_b),
                    "trackable": o is not None,
                    "conflict_truth": tc,
                    "conflict_observed": oc,
                    "ttc_truth_s": tttc,
                    "ttc_observed_s": o[1] if o else np.inf,
                    "separation_truth_m": trng,
                    "separation_vert_truth_m": td_v,
                    "separation_observed_m": o[4] if o else np.nan,
                    "closing_speed_truth_mps": tclose,
                    "d_cpa_truth_m": td_h,
                    "d_cpa_observed_m": o[2] if o else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["n_true_negative"] = n_true_negative
    df.attrs["n_pair_observations"] = n_evaluated
    df.attrs["record_mode"] = record
    if df.empty:
        return df

    for k, v in config.describe().items():
        if k != "scenario_id":
            df[k] = v

    both = np.isfinite(df["ttc_truth_s"]) & np.isfinite(df["ttc_observed_s"])
    df["ttc_error_signed_s"] = np.where(both, df["ttc_truth_s"] - df["ttc_observed_s"], np.nan)
    df["ttc_error_abs_s"] = df["ttc_error_signed_s"].abs()
    return df
