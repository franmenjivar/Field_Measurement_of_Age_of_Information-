"""Replay a real surveillance stream against real ground truth.

Everywhere else in this study the observed layer is produced by the telemetry
emulator.  Here it is not emulated at all: the observed states are the tracks an
operational UTM actually held, carrying the information age that system actually
delivered.  Ground truth is the aircraft's own onboard telemetry.

That makes this the only experiment in which every element of the chain --
trajectory, surveillance, and information age -- is measured rather than
modelled.

**Deliberately threshold-free.** A conflict/no-conflict verdict is not produced.
In the BUBBLES campaign a 30 m separation criterion is dominated by co-located
loitering operations (aircraft holding station near one another at ~1-2 m/s
relative speed for a third of a flight), which is a different phenomenon from a
tactical encounter and would swamp any classification metric.  What is measured
instead is the *estimation error* -- CPA distance and time-to-CPA -- restricted
to genuinely converging geometry.  Selecting on closing speed also removes the
co-located pairs organically, since formation-like pairs have almost no closing
speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..conflict.cpa import compute_cpa
from ..core.schema import State
from ..surveillance.state_store import SurveillanceStateStore
from ..telemetry.generator import TrajectorySampler
from ..telemetry.message import TelemetryMessage


@dataclass(frozen=True)
class ReplayConfig:
    """Selection criteria defining which real geometry is analysed."""

    #: only consider pairs closer than this (tactical relevance)
    max_range_m: float = 500.0
    #: only consider epochs where the pair is genuinely converging
    min_closing_speed_mps: float = 3.0
    #: CPA search horizon
    horizon_s: float = 30.0
    #: decision cadence
    decision_rate_hz: float = 1.0
    #: ignore an aircraft whose freshest track is older than this (dropped track)
    max_usable_aoi_s: float = 30.0
    #: require both aircraft to be airborne
    min_speed_mps: float = 1.0
    #: dead-reckon each observed state forward by its own measured age before use
    extrapolate: bool = False

    def describe(self) -> dict:
        return {
            "max_range_m": self.max_range_m,
            "min_closing_speed_mps": self.min_closing_speed_mps,
            "horizon_s": self.horizon_s,
            "decision_rate_hz": self.decision_rate_hz,
            "max_usable_aoi_s": self.max_usable_aoi_s,
            "min_speed_mps": self.min_speed_mps,
            "extrapolate": self.extrapolate,
        }


def store_from_observations(observations: pd.DataFrame) -> SurveillanceStateStore:
    """Build a surveillance store from real UTM track rows.

    ``t_gen_s`` (the system's own ``t_last``) becomes the generation time and the
    cycle time becomes the arrival time, so the store reproduces the *measured*
    information age rather than a modelled one.
    """
    store = SurveillanceStateStore()
    msgs = []
    d = observations.dropna(subset=["t_s", "t_gen_s", "x_m", "y_m", "uas_id"])
    for r in d.itertuples():
        msgs.append(
            TelemetryMessage(
                uas_id=str(r.uas_id),
                t_gen_s=float(r.t_gen_s),
                t_arrive_s=float(r.t_s),
                state=State(
                    uas_id=str(r.uas_id),
                    t_s=float(r.t_gen_s),
                    x_m=float(r.x_m),
                    y_m=float(r.y_m),
                    z_m=float(r.z_m) if np.isfinite(r.z_m) else 0.0,
                    vx_mps=float(r.vx_mps) if np.isfinite(r.vx_mps) else 0.0,
                    vy_mps=float(r.vy_mps) if np.isfinite(r.vy_mps) else 0.0,
                    vz_mps=0.0,
                ),
            )
        )
    store.ingest(msgs)
    return store


def replay_scenario(
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    config: ReplayConfig | None = None,
    *,
    scenario_id: str = "replay",
) -> pd.DataFrame:
    """Compare truth-derived and surveillance-derived CPA/TTC, pair by pair.

    Returns one row per (decision epoch, converging pair) with the geometry as
    the truth says it was, the geometry the UTM would have computed from what it
    held, and the measured age of that held state.
    """
    cfg = config or ReplayConfig()

    samplers = {
        uid: TrajectorySampler(g)
        for uid, g in truth.groupby("uas_id", sort=True)
        if len(g) >= 10
    }
    obs = observations[observations["uas_id"].isin(samplers)]
    if obs.empty or len(samplers) < 2:
        return pd.DataFrame()

    store = store_from_observations(obs)

    t0 = max(min(s.t_start for s in samplers.values()), float(obs["t_s"].min()))
    t1 = min(max(s.t_end for s in samplers.values()), float(obs["t_s"].max()))
    if t1 <= t0:
        return pd.DataFrame()

    ids = sorted(samplers)
    rows = []

    for t in np.arange(t0, t1, 1.0 / cfg.decision_rate_hz):
        t = float(t)

        truth_states, observed_states, ages = {}, {}, {}
        for uid in ids:
            s = samplers[uid]
            if not (s.t_start <= t <= s.t_end):
                continue
            st = s.at(t)
            if st.speed_mps < cfg.min_speed_mps:
                continue          # on the ground or holding: not tactical traffic
            truth_states[uid] = st

            msg = store.latest(uid, t)
            if msg is None:
                continue
            age = t - msg.t_gen_s
            if not (0.0 <= age <= cfg.max_usable_aoi_s):
                continue          # track dropped; the UTM has no usable state
            ages[uid] = age
            # Dead reckoning uses the velocity the UTM itself reported, which
            # carries its own error -- that is the point of the comparison.
            observed_states[uid] = msg.state.propagate(age) if cfg.extrapolate else msg.state

        common = sorted(set(truth_states) & set(observed_states))
        for i, a in enumerate(common):
            for b in common[i + 1 :]:
                ta, tb = truth_states[a], truth_states[b]
                true_cpa = compute_cpa(ta, tb, horizon_s=cfg.horizon_s)

                # Selection is made on TRUTH geometry, so the observed layer
                # cannot influence which situations enter the sample.
                if true_cpa.range_now_m > cfg.max_range_m:
                    continue
                if true_cpa.closing_speed_mps < cfg.min_closing_speed_mps:
                    continue

                oa, ob = observed_states[a], observed_states[b]
                obs_cpa = compute_cpa(oa, ob, horizon_s=cfg.horizon_s)

                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "t_s": t,
                        "uas_a": a,
                        "uas_b": b,
                        "same_operator": a.split("-")[0] == b.split("-")[0],
                        "aoi_a_s": ages[a],
                        "aoi_b_s": ages[b],
                        "aoi_s": max(ages[a], ages[b]),
                        "range_truth_m": true_cpa.range_now_m,
                        "range_observed_m": obs_cpa.range_now_m,
                        "closing_truth_mps": true_cpa.closing_speed_mps,
                        "closing_observed_mps": obs_cpa.closing_speed_mps,
                        "t_cpa_truth_s": true_cpa.t_cpa_s,
                        "t_cpa_observed_s": obs_cpa.t_cpa_s,
                        "d_cpa_truth_m": true_cpa.d_cpa_horiz_m,
                        "d_cpa_observed_m": obs_cpa.d_cpa_horiz_m,
                        "observed_diverging": obs_cpa.diverging,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sign convention matches the rest of the study: negative t_cpa error means
    # the UTM believed the closest approach was further away in time than it was.
    df["t_cpa_error_s"] = df["t_cpa_truth_s"] - df["t_cpa_observed_s"]
    df["t_cpa_overestimated"] = df["t_cpa_error_s"] < 0
    df["d_cpa_error_m"] = df["d_cpa_observed_m"] - df["d_cpa_truth_m"]
    # Positive d_cpa error = the UTM thought the aircraft would pass further
    # apart than they actually would: the optimistic, dangerous direction.
    df["d_cpa_optimistic"] = df["d_cpa_error_m"] > 0
    df["range_error_m"] = df["range_observed_m"] - df["range_truth_m"]

    for k, v in cfg.describe().items():
        df[k] = v
    return df
