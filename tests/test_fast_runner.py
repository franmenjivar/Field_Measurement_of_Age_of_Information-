"""The vectorised runner must not change any conclusion, only the runtime."""

from __future__ import annotations

import numpy as np
import pytest

from telemetry_freshness.conflict.detector import ConflictCriteria
from telemetry_freshness.evaluation.metrics import detection_counts
from telemetry_freshness.experiments.fast_runner import run_scenario_fast
from telemetry_freshness.experiments.runner import ScenarioConfig, run_scenario
from telemetry_freshness.simulation.encounters import EncounterSpec, synthetic_encounter
from telemetry_freshness.telemetry.network_model import TelemetryConfig

GEOMETRIES = ["head_on", "crossing", "converging", "overtaking"]
COMPARED = [
    "conflict_truth",
    "conflict_observed",
    "ttc_truth_s",
    "ttc_observed_s",
    "d_cpa_truth_m",
    "aoi_s",
    "separation_truth_m",
]


def _run(geom: str):
    traj = synthetic_encounter(EncounterSpec(geom, lead_in_s=45, duration_s=90))
    cfg = ScenarioConfig(
        geom,
        TelemetryConfig(2.0, latency_ms=250),
        criteria=ConflictCriteria(30.0, 15.0, 30.0),
        decision_rate_hz=5,
        seed=11,
    )
    return run_scenario(traj, cfg), run_scenario_fast(traj, cfg)


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_fast_runner_matches_reference_on_retained_pairs(geom):
    ref, fast = _run(geom)
    merged = ref.merge(
        fast, on=["t_s", "uas_a", "uas_b"], how="left", suffixes=("_ref", "_fast"),
        indicator=True,
    )
    kept = merged[merged["_merge"] == "both"]
    assert len(kept) > 0
    for col in COMPARED:
        assert np.allclose(
            kept[f"{col}_ref"].astype(float),
            kept[f"{col}_fast"].astype(float),
            equal_nan=True,
        ), col


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_range_screening_only_discards_true_negatives(geom):
    """A screened-out pair must never have been a conflict in either layer."""
    ref, fast = _run(geom)
    merged = ref.merge(
        fast, on=["t_s", "uas_a", "uas_b"], how="left", suffixes=("_ref", "_fast"),
        indicator=True,
    )
    dropped = merged[merged["_merge"] == "left_only"]
    assert not dropped["conflict_truth_ref"].any()
    assert not dropped["conflict_observed_ref"].any()


def test_relevant_record_mode_preserves_detection_metrics():
    traj = synthetic_encounter(EncounterSpec("crossing", lead_in_s=45, duration_s=90))
    cfg = ScenarioConfig(
        "x", TelemetryConfig(2.0, latency_ms=250),
        criteria=ConflictCriteria(30.0, 15.0, 30.0), decision_rate_hz=5, seed=11,
    )
    full = run_scenario_fast(traj, cfg, record="all")
    lean = run_scenario_fast(traj, cfg, record="relevant")

    a = detection_counts(full.conflict_truth, full.conflict_observed)
    b = detection_counts(lean.conflict_truth, lean.conflict_observed)
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
    assert a.recall == pytest.approx(b.recall)
    assert a.precision == pytest.approx(b.precision)
    # The discarded rows are accounted for, not lost.
    assert lean.attrs["n_true_negative"] == a.tn
    assert len(lean) < len(full)
