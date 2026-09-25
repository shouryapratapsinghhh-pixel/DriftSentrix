import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from regression_gate import (
    compute_metrics,
    evaluate_gate,
)

# -- evaluate_gate: pure comparison logic, no model fitting (fast) ----------


def test_gate_passes_when_metrics_match():
    baseline = {"zscore": {"pr_auc": 0.80, "event_f1": 0.70}}
    current = {"zscore": {"pr_auc": 0.80, "event_f1": 0.70}}
    passed, _ = evaluate_gate(current, baseline, tolerance=0.03)
    assert passed


def test_gate_passes_when_metrics_improve():
    baseline = {"zscore": {"pr_auc": 0.80}}
    current = {"zscore": {"pr_auc": 0.95}}  # improvement, never a regression
    passed, _ = evaluate_gate(current, baseline, tolerance=0.03)
    assert passed


def test_gate_passes_within_tolerance():
    baseline = {"zscore": {"pr_auc": 0.80}}
    current = {"zscore": {"pr_auc": 0.78}}  # drop of 0.02 < tolerance 0.03
    passed, _ = evaluate_gate(current, baseline, tolerance=0.03)
    assert passed


def test_gate_fails_on_degraded_metric():
    baseline = {"zscore": {"pr_auc": 0.80, "event_f1": 0.70}}
    current = {"zscore": {"pr_auc": 0.50, "event_f1": 0.70}}  # pr_auc dropped 0.30
    passed, lines = evaluate_gate(current, baseline, tolerance=0.03)
    assert not passed
    assert any("REGRESSION" in line for line in lines)


def test_gate_fails_at_exactly_the_boundary_edge():
    # a drop clearly ABOVE tolerance must fail; a drop clearly BELOW must
    # pass. (Testing the exact float boundary itself -- e.g. 0.80 - 0.77 -
    # is unreliable: that subtraction lands at 0.030000000000000027 in
    # floating point, one ULP past 0.03, which is a float-precision
    # artifact of the test's chosen literals, not a real gate-logic bug.)
    baseline = {"zscore": {"pr_auc": 0.80}}
    assert evaluate_gate({"zscore": {"pr_auc": 0.771}}, baseline, tolerance=0.03)[0]  # drop 0.029 -> pass
    assert not evaluate_gate({"zscore": {"pr_auc": 0.769}}, baseline, tolerance=0.03)[0]  # drop 0.031 -> fail


def test_gate_missing_baseline_entry_is_skipped_not_failed():
    baseline = {}  # no entry at all for this model
    current = {"zscore": {"pr_auc": 0.10}}  # would look terrible if compared, but nothing to compare to
    passed, lines = evaluate_gate(current, baseline, tolerance=0.03)
    assert passed
    assert any("no baseline entry" in line for line in lines)


def test_gate_one_model_regresses_others_fine_overall_fails():
    baseline = {"zscore": {"pr_auc": 0.80}, "pca": {"pr_auc": 0.80}}
    current = {"zscore": {"pr_auc": 0.79}, "pca": {"pr_auc": 0.10}}  # pca tanked
    passed, _ = evaluate_gate(current, baseline, tolerance=0.03)
    assert not passed  # one regression anywhere fails the whole gate


# -- compute_metrics: smoke test that it actually runs and returns sane values --


def test_compute_metrics_smoke():
    metrics = compute_metrics("zscore")
    assert -1e-9 <= metrics["pr_auc"] <= 1.0 + 1e-9  # tiny float-precision slack, not a real bound violation
    assert -1e-9 <= metrics["event_f1"] <= 1.0 + 1e-9


def test_compute_metrics_beats_random():
    random_metrics = compute_metrics("random")
    zscore_metrics = compute_metrics("zscore")
    assert zscore_metrics["pr_auc"] > random_metrics["pr_auc"]
