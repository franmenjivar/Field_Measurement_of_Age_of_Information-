"""The central scientific control: the observed path cannot reach ground truth.

If the conflict detector could see the true state -- through a back-reference on
the store, the message or the state object -- then any difference between the
truth and observed verdicts would be uninterpretable.  These tests fail loudly
if such a path is ever introduced.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from telemetry_freshness.conflict.detector import ConflictDetector
from telemetry_freshness.core.rng import make_rng
from telemetry_freshness.simulation.encounters import EncounterSpec, synthetic_encounter
from telemetry_freshness.surveillance.state_store import SurveillanceStateStore
from telemetry_freshness.telemetry.generator import TrajectorySampler, generate_stream
from telemetry_freshness.telemetry.network_model import TelemetryConfig


@pytest.fixture
def store_and_sampler():
    traj = synthetic_encounter(EncounterSpec("head_on", lead_in_s=30, duration_s=60))
    g = traj[traj.uas_id == "UAS-A"]
    sampler = TrajectorySampler(g)
    store = SurveillanceStateStore()
    store.ingest(
        generate_stream(
            sampler,
            TelemetryConfig(update_rate_hz=1.0, latency_ms=500),
            make_rng(1, "test"),
        )
    )
    return store, sampler, traj


def test_store_holds_no_trajectory_reference(store_and_sampler):
    store, sampler, _ = store_and_sampler
    reachable = {id(o) for o in gc.get_referents(store, *gc.get_referents(store))}
    assert id(sampler) not in reachable
    for msg in store._arrivals["UAS-A"]:
        assert not hasattr(msg.state, "sampler")
        assert not hasattr(msg, "truth")
        # The state must be a plain immutable value object.
        with pytest.raises((AttributeError, TypeError)):
            msg.state.x_m = 0.0


def test_detector_sees_only_stale_state(store_and_sampler):
    store, sampler, _ = store_and_sampler
    t = 40.0
    observed = store.observed_state("UAS-A", t)
    truth = sampler.at(t)
    assert observed is not None
    # With 1 Hz sampling and 500 ms latency the observation must be behind.
    assert observed.t_s < truth.t_s
    assert abs(observed.x_m - truth.x_m) > 1.0


def test_aoi_is_nan_before_first_arrival(store_and_sampler):
    store, _, _ = store_and_sampler
    assert np.isnan(store.aoi("UAS-A", -1.0))
    assert store.latest("UAS-A", -1.0) is None
    # Untracked must never be reported as a zero age.
    assert store.aoi("UAS-A", -1.0) != 0.0


def test_same_detector_both_paths():
    """Truth and observed verdicts must come from the same detector class."""
    traj = synthetic_encounter(EncounterSpec("crossing", lead_in_s=30, duration_s=60))
    det = ConflictDetector()
    samplers = {u: TrajectorySampler(g) for u, g in traj.groupby("uas_id")}
    store = SurveillanceStateStore()
    for u, s in samplers.items():
        store.ingest(
            generate_stream(s, TelemetryConfig(2.0, latency_ms=0.0), make_rng(2, u))
        )
    t = 30.0
    truth_states = {u: s.at(t) for u, s in samplers.items()}
    obs_states = store.snapshot(t)
    # Same object, two inputs -- not two different implementations.
    assert det.assess(truth_states, t)[0].uas_a == det.assess(obs_states, t)[0].uas_a
