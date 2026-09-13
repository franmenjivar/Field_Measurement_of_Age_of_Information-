"""End-to-end experiment execution: truth in, paired records out.

One function does the whole pipeline for a scenario: sample the ground truth
into a delayed telemetry stream, let the UTM store age, then run the *same*
detector twice -- once on truth, once on the store -- and emit one record per
(decision time, pair) with both verdicts side by side.

The truth detector and the observed detector are separate objects fed from
separate sources; nothing downstream of :func:`run_scenario` can reach the
trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..conflict.detector import ConflictCriteria, ConflictDetector
from ..core.rng import make_rng
from ..core.schema import State
from ..surveillance.state_store import SurveillanceStateStore
from ..telemetry.generator import TrajectorySampler, generate_stream
from ..telemetry.network_model import TelemetryConfig


@dataclass
class ScenarioConfig:
    """Everything needed to reproduce one scenario run."""

    scenario_id: str
    telemetry: TelemetryConfig
    criteria: ConflictCriteria = field(default_factory=ConflictCriteria)
    decision_rate_hz: float = 5.0
    seed: int = 42
    t_start: float | None = None
    t_end: float | None = None
    extrapolate: bool = False
    keep_newest_generation: bool = True

    def describe(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "update_rate_hz": self.telemetry.update_rate_hz,
            "latency_ms": self.telemetry.latency_ms,
            "jitter_ms": self.telemetry.jitter_ms,
            "packet_loss": self.telemetry.packet_loss_probability,
            "latency_model": self.telemetry.latency_model,
            "horizontal_separation_m": self.criteria.horizontal_separation_m,
            "vertical_separation_m": self.criteria.vertical_separation_m,
            "horizon_s": self.criteria.horizon_s,
            "decision_rate_hz": self.decision_rate_hz,
            "seed": self.seed,
            "extrapolate": self.extrapolate,
            "keep_newest_generation": self.keep_newest_generation,
        }


def _pair_aoi(age_a: float | None, age_b: float | None) -> float:
    """Pair AoI = the stalest of the two; nan if either is untracked."""
    if age_a is None or age_b is None:
        return np.nan
    return float(max(age_a, age_b))


def _separation(a: State, b: State) -> tuple[float, float]:
    return float(np.hypot(a.x_m - b.x_m, a.y_m - b.y_m)), float(abs(a.z_m - b.z_m))


def run_scenario(trajectories: pd.DataFrame, config: ScenarioConfig) -> pd.DataFrame:
    """Run one scenario and return per-(time, pair) paired verdicts.

    Parameters
    ----------
    trajectories
        Unified-schema ground truth for every aircraft in the scenario.
    """
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

    # Each aircraft gets its own named RNG stream so adding or removing an
    # aircraft cannot perturb another's jitter draw.
    store = SurveillanceStateStore(keep_newest_generation=config.keep_newest_generation)
    for uid, s in samplers.items():
        rng = make_rng(config.seed, f"{config.scenario_id}|{uid}")
        store.ingest(generate_stream(s, config.telemetry, rng, t_start=t0, t_end=t1))

    detector = ConflictDetector(config.criteria)
    decision_times = np.arange(t0, t1, 1.0 / config.decision_rate_hz)

    rows = []
    for t in decision_times:
        t = float(t)
        truth_states = {uid: s.at(t) for uid, s in samplers.items()}

        observed_states: dict[str, State] = {}
        ages: dict[str, float] = {}
        for uid in samplers:
            msg = store.latest(uid, t)
            if msg is None:
                continue
            age = t - msg.t_gen_s
            ages[uid] = age
            observed_states[uid] = msg.state.propagate(age) if config.extrapolate else msg.state

        truth_pairs = {(p.uas_a, p.uas_b): p for p in detector.assess(truth_states, t)}
        obs_pairs = {(p.uas_a, p.uas_b): p for p in detector.assess(observed_states, t)}

        for key, tp in truth_pairs.items():
            a, b = key
            op = obs_pairs.get(key)
            sep_h, sep_v = _separation(truth_states[a], truth_states[b])

            if op is None:
                # At least one aircraft is untracked: the UTM cannot assess this
                # pair at all.  That is a distinct failure from "assessed wrong".
                obs_sep_h = np.nan
                obs_conflict = False
                obs_ttc = np.inf
                trackable = False
            else:
                obs_sep_h = op.range_m
                obs_conflict = op.conflict
                obs_ttc = op.ttc_s
                trackable = True

            rows.append(
                {
                    "scenario_id": config.scenario_id,
                    "t_s": t,
                    "uas_a": a,
                    "uas_b": b,
                    "aoi_a_s": ages.get(a, np.nan),
                    "aoi_b_s": ages.get(b, np.nan),
                    # The pair is only as fresh as its stalest aircraft.
                    "aoi_s": _pair_aoi(ages.get(a), ages.get(b)),
                    "trackable": trackable,
                    "conflict_truth": tp.conflict,
                    "conflict_observed": obs_conflict,
                    "ttc_truth_s": tp.ttc_s,
                    "ttc_observed_s": obs_ttc,
                    "separation_truth_m": sep_h,
                    "separation_vert_truth_m": sep_v,
                    "separation_observed_m": obs_sep_h,
                    "closing_speed_truth_mps": tp.closing_speed_mps,
                    "d_cpa_truth_m": tp.d_cpa_horiz_m,
                    "d_cpa_observed_m": op.d_cpa_horiz_m if op else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for k, v in config.describe().items():
        if k != "scenario_id":
            df[k] = v

    # Signed error, positive = observed TTC longer than reality (dangerous).
    both = np.isfinite(df["ttc_truth_s"]) & np.isfinite(df["ttc_observed_s"])
    df["ttc_error_signed_s"] = np.where(both, df["ttc_truth_s"] - df["ttc_observed_s"], np.nan)
    df["ttc_error_abs_s"] = df["ttc_error_signed_s"].abs()
    return df
