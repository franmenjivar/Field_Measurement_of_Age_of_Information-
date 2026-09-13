"""Age-of-Information measurement and freshness-induced displacement.

AoI_i(t) = t - u_i(t), with u_i the generation timestamp of the freshest state
the UTM holds for aircraft i (research plan, section 3.4).

Two displacement quantities are reported and they are not the same thing:

``stale_displacement_m``
    the *measured* distance between where the aircraft actually was and the
    position the UTM was working from.  This is the real error.

``predicted_displacement_m``
    ``v * AoI``, the bound the research plan writes down (section 7).  Reporting
    both shows how well that bound holds during real manoeuvres -- where a
    constant-velocity argument breaks down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.schema import State
from ..telemetry.generator import TrajectorySampler
from .state_store import SurveillanceStateStore


def aoi_series(
    store: SurveillanceStateStore, uas_id: str, decision_times: np.ndarray
) -> np.ndarray:
    """AoI at each decision time; ``nan`` where the aircraft is untracked."""
    return np.array([store.aoi(uas_id, float(t)) for t in decision_times])


def freshness_errors(
    store: SurveillanceStateStore,
    sampler: TrajectorySampler,
    decision_times: np.ndarray,
    *,
    extrapolate: bool = False,
) -> pd.DataFrame:
    """Compare truth against the UTM's view at each decision time.

    Parameters
    ----------
    extrapolate
        When true the observed state is dead-reckoned forward by its own age
        before comparison, which is what a UTM that trusts reported velocity
        would do.  Reporting both settings separates "the data is old" from
        "the data is old and nobody compensated".
    """
    uid = sampler.uas_id
    rows = []
    for t in np.asarray(decision_times, dtype=float):
        msg = store.latest(uid, float(t))
        truth: State = sampler.at(float(t))
        if msg is None:
            rows.append(
                {
                    "uas_id": uid,
                    "t_s": t,
                    "aoi_s": np.nan,
                    "tracked": False,
                    "pos_error_m": np.nan,
                    "horiz_error_m": np.nan,
                    "vert_error_m": np.nan,
                    "vel_error_mps": np.nan,
                    "heading_error_deg": np.nan,
                    "predicted_displacement_m": np.nan,
                    "truth_speed_mps": truth.speed_mps,
                }
            )
            continue

        aoi = float(t) - msg.t_gen_s
        obs = msg.state.propagate(aoi) if extrapolate else msg.state

        dp = truth.position - obs.position
        dv = truth.velocity - obs.velocity
        h_truth = np.degrees(np.arctan2(truth.vx_mps, truth.vy_mps)) % 360.0
        h_obs = np.degrees(np.arctan2(obs.vx_mps, obs.vy_mps)) % 360.0
        dh = (h_truth - h_obs + 180.0) % 360.0 - 180.0

        rows.append(
            {
                "uas_id": uid,
                "t_s": float(t),
                "aoi_s": aoi,
                "tracked": True,
                "pos_error_m": float(np.linalg.norm(dp)),
                "horiz_error_m": float(np.hypot(dp[0], dp[1])),
                "vert_error_m": float(abs(dp[2])),
                "vel_error_mps": float(np.linalg.norm(dv)),
                "heading_error_deg": float(abs(dh)),
                # v*AoI using the speed the UTM knows about -- the bound a real
                # system could actually compute at decision time.
                "predicted_displacement_m": float(msg.state.speed_mps * aoi),
                "truth_speed_mps": truth.speed_mps,
            }
        )
    return pd.DataFrame(rows)


def summarise_aoi(aoi_s: np.ndarray | pd.Series) -> dict[str, float]:
    """Distribution summary; the tail matters more than the mean for safety."""
    a = pd.Series(aoi_s).dropna().to_numpy(dtype=float)
    if a.size == 0:
        return dict.fromkeys(
            ("n", "mean", "median", "p95", "p99", "max"), float("nan")
        ) | {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }
