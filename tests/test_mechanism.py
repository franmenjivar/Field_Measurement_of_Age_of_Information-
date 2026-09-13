"""Analytic checks on AoI, CPA and the freshness-error relationship."""

from __future__ import annotations

import numpy as np
import pytest

from telemetry_freshness.conflict.cpa import compute_cpa
from telemetry_freshness.conflict.detector import ConflictCriteria, ConflictDetector
from telemetry_freshness.core.rng import make_rng
from telemetry_freshness.core.schema import State
from telemetry_freshness.evaluation.metrics import detection_counts, ttc_error
from telemetry_freshness.evaluation.statistics import wilson_interval
from telemetry_freshness.simulation.encounters import EncounterSpec, synthetic_encounter
from telemetry_freshness.surveillance.state_store import SurveillanceStateStore
from telemetry_freshness.telemetry.generator import TrajectorySampler, generate_stream
from telemetry_freshness.telemetry.network_model import TelemetryConfig


def test_cpa_head_on_closed_form():
    a = State("A", 0, -500, 0, 100, 15, 0, 0)
    b = State("B", 0, 500, 0, 100, -15, 0, 0)
    c = compute_cpa(a, b, horizon_s=120)
    assert c.t_cpa_s == pytest.approx(1000 / 30, rel=1e-9)
    assert c.d_cpa_m == pytest.approx(0.0, abs=1e-6)
    assert c.closing_speed_mps == pytest.approx(30.0)


def test_cpa_miss_distance_preserved():
    a = State("A", 0, -500, 0, 100, 15, 0, 0)
    b = State("B", 0, 500, 40, 100, -15, 0, 0)
    c = compute_cpa(a, b, horizon_s=120)
    assert c.d_cpa_horiz_m == pytest.approx(40.0, abs=1e-6)


def test_diverging_pair_is_not_a_conflict():
    a = State("A", 0, 0, 0, 100, -10, 0, 0)
    b = State("B", 0, 20, 0, 100, 10, 0, 0)
    det = ConflictDetector(ConflictCriteria(horizontal_separation_m=30))
    p = det.assess_pair(a, b, 0.0)
    assert not p.conflict
    assert np.isinf(p.ttc_s)


def test_aoi_sawtooth_matches_sampling_interval():
    """With zero latency, AoI must sweep 0 -> 1/rate and reset."""
    traj = synthetic_encounter(EncounterSpec("head_on", lead_in_s=30, duration_s=60))
    s = TrajectorySampler(traj[traj.uas_id == "UAS-A"])
    cfg = TelemetryConfig(update_rate_hz=2.0, latency_ms=0.0, randomise_phase=False)
    store = SurveillanceStateStore()
    store.ingest(generate_stream(s, cfg, make_rng(0, "x")))

    t = np.arange(5.0, 50.0, 0.01)
    aoi = np.array([store.aoi("UAS-A", float(x)) for x in t])
    assert np.nanmin(aoi) >= 0.0
    assert np.nanmax(aoi) == pytest.approx(0.5, abs=0.02)
    assert np.nanmean(aoi) == pytest.approx(0.25, abs=0.02)


def test_added_latency_shifts_aoi_by_exactly_that_amount():
    traj = synthetic_encounter(EncounterSpec("head_on", lead_in_s=30, duration_s=60))
    s = TrajectorySampler(traj[traj.uas_id == "UAS-A"])
    t = np.arange(10.0, 50.0, 0.01)

    def mean_aoi(latency_ms):
        store = SurveillanceStateStore()
        store.ingest(
            generate_stream(
                s,
                TelemetryConfig(2.0, latency_ms=latency_ms, randomise_phase=False),
                make_rng(0, "x"),
            )
        )
        return np.nanmean([store.aoi("UAS-A", float(x)) for x in t])

    assert mean_aoi(500) - mean_aoi(0) == pytest.approx(0.5, abs=0.02)


def test_stale_displacement_tracks_speed_times_aoi():
    """For constant velocity, measured displacement must equal v*AoI exactly."""
    traj = synthetic_encounter(
        EncounterSpec("head_on", speed_a_mps=20.0, lead_in_s=30, duration_s=60)
    )
    s = TrajectorySampler(traj[traj.uas_id == "UAS-A"])
    store = SurveillanceStateStore()
    store.ingest(
        generate_stream(
            s, TelemetryConfig(1.0, latency_ms=0.0, randomise_phase=False), make_rng(0, "x")
        )
    )
    for t in (12.34, 25.7, 41.9):
        msg = store.latest("UAS-A", t)
        aoi = t - msg.t_gen_s
        truth = s.at(t)
        measured = np.linalg.norm(truth.position - msg.state.position)
        assert measured == pytest.approx(20.0 * aoi, rel=1e-6)


def test_reordered_message_does_not_displace_fresher_state():
    traj = synthetic_encounter(EncounterSpec("head_on", lead_in_s=30, duration_s=60))
    s = TrajectorySampler(traj[traj.uas_id == "UAS-A"])
    cfg = TelemetryConfig(2.0, latency_ms=300, jitter_ms=400, latency_model="fixed")
    store = SurveillanceStateStore(keep_newest_generation=True)
    store.ingest(generate_stream(s, cfg, make_rng(7, "x")))
    gens = [
        store.latest("UAS-A", float(t)).t_gen_s
        for t in np.arange(10, 50, 0.05)
        if store.latest("UAS-A", float(t)) is not None
    ]
    assert all(b >= a for a, b in zip(gens, gens[1:], strict=False))


def test_ttc_error_sign_convention():
    """Research plan section 8: truth 5.2 s, observed 8.7 s is the DANGEROUS case."""
    df = ttc_error(np.array([5.2]), np.array([8.7]))
    assert df["ttc_error_signed_s"].iloc[0] == pytest.approx(-3.5)
    assert df["ttc_error_abs_s"].iloc[0] == pytest.approx(3.5)
    # Observed TTC exceeds reality -> the system believes it has more time.
    assert bool(df["ttc_overestimated"].iloc[0]) is True

    # And the opposite direction must not be flagged as dangerous.
    safe = ttc_error(np.array([8.7]), np.array([5.2]))
    assert safe["ttc_error_signed_s"].iloc[0] == pytest.approx(3.5)
    assert bool(safe["ttc_overestimated"].iloc[0]) is False


def test_detection_counts_and_wilson():
    c = detection_counts([True, True, False, False], [True, False, True, False])
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 1, 1, 1)
    assert c.recall == pytest.approx(0.5)
    lo, hi = wilson_interval(99, 100)
    assert 0.94 < lo < 0.995 and hi <= 1.0


def test_packet_loss_reduces_message_count():
    traj = synthetic_encounter(EncounterSpec("head_on", lead_in_s=30, duration_s=120))
    s = TrajectorySampler(traj[traj.uas_id == "UAS-A"])
    full = generate_stream(s, TelemetryConfig(5.0), make_rng(3, "a"))
    lossy = generate_stream(
        s, TelemetryConfig(5.0, packet_loss_probability=0.3), make_rng(3, "a")
    )
    assert 0.6 < len(lossy) / len(full) < 0.8


def test_run_is_reproducible_from_seed():
    from telemetry_freshness.experiments.runner import ScenarioConfig, run_scenario

    traj = synthetic_encounter(EncounterSpec("crossing", lead_in_s=30, duration_s=60))
    cfg = ScenarioConfig(
        "T", TelemetryConfig(2.0, latency_ms=250, jitter_ms=50), seed=99, decision_rate_hz=5
    )
    a = run_scenario(traj, cfg)
    b = run_scenario(traj, cfg)
    assert a.equals(b)
