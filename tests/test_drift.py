import numpy as np
import pytest
from scipy.stats import ks_2samp

from anomaly.monitor.drift import DriftMonitor, ks_critical_value, ks_statistic, psi

# -- ks_statistic: matches scipy ---------------------------------------------


def test_ks_statistic_matches_scipy_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    ours = ks_statistic(a, b)
    scipy_result = ks_2samp(a, b)
    assert ours == pytest.approx(scipy_result.statistic, abs=1e-9)


def test_ks_statistic_matches_scipy_shifted_distribution():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0, size=300)
    b = rng.normal(loc=2, size=300)
    ours = ks_statistic(a, b)
    scipy_result = ks_2samp(a, b)
    assert ours == pytest.approx(scipy_result.statistic, abs=1e-9)


def test_ks_statistic_identical_samples_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert ks_statistic(a, a) == 0.0


# -- psi ----------------------------------------------------------------------


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=2000)
    cur = rng.normal(size=1000)  # same distribution, different draw
    assert psi(ref, cur) < 0.05  # well under the 0.1 warn threshold


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(3)
    ref = rng.normal(loc=0, scale=1, size=2000)
    cur = rng.normal(loc=5, scale=1, size=1000)  # totally disjoint
    assert psi(ref, cur) > 0.25  # well past the default alert threshold


def test_psi_constant_reference_no_movement():
    ref = np.full(500, 3.0)
    cur = np.full(200, 3.0)
    assert psi(ref, cur) == 0.0


def test_psi_constant_reference_with_movement():
    ref = np.full(500, 3.0)
    cur = np.full(200, 8.0)
    assert psi(ref, cur) > 0.25


# -- DriftMonitor ---------------------------------------------------------


def test_no_alert_on_same_distribution():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(1000, 3))
    monitor = DriftMonitor(reference, window=200)
    for _ in range(200):
        monitor.update(rng.normal(size=3))
    report = monitor.report()
    assert report["status"] == "ok"


def test_alert_on_mean_shift():
    rng = np.random.default_rng(5)
    reference = rng.normal(loc=0, size=(1000, 3))
    monitor = DriftMonitor(reference, window=200)
    for _ in range(200):
        monitor.update(rng.normal(loc=5, size=3))  # large mean shift on every feature
    report = monitor.report()
    assert report["status"] == "alert"
    assert len(report["top_drifted_features"]) > 0


def test_alert_on_variance_shift():
    rng = np.random.default_rng(6)
    reference = rng.normal(loc=0, scale=1, size=(1000, 2))
    monitor = DriftMonitor(reference, window=300)
    for _ in range(300):
        monitor.update(rng.normal(loc=0, scale=8, size=2))  # variance blows up, mean unchanged
    report = monitor.report()
    assert report["status"] in ("warn", "alert")  # KS is sensitive to spread even with matched means


def test_insufficient_data_does_not_crash():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(500, 2))
    monitor = DriftMonitor(reference, window=100)
    monitor.update(rng.normal(size=2))  # only one point -- window not yet full
    report = monitor.report()
    assert report["status"] == "insufficient_data"
    assert report["features"] == []


def test_constant_feature_does_not_crash():
    rng = np.random.default_rng(8)
    reference = np.column_stack([np.full(500, 1.0), rng.normal(size=500)])
    monitor = DriftMonitor(reference, window=100)
    for _ in range(100):
        monitor.update([1.0, rng.normal()])  # constant feature stays constant
    report = monitor.report()
    assert report["status"] in ("ok", "warn", "alert")  # just must not crash
    assert report["features"][0]["psi"] == 0.0  # the constant feature shows no drift


def test_score_distribution_mode_1d_reference():
    """Passing a 1D array (e.g. anomaly scores, not raw features) works
    identically -- this IS the "monitor the score distribution" mode.
    """
    rng = np.random.default_rng(9)
    reference_scores = rng.exponential(scale=1.0, size=1000)
    monitor = DriftMonitor(reference_scores, window=200)
    assert monitor.n_features == 1
    for _ in range(200):
        monitor.update(rng.exponential(scale=1.0))
    report = monitor.report()
    assert report["status"] == "ok"


def test_ks_critical_value_decreases_with_more_samples():
    # more data -> tighter critical value (easier to detect real drift)
    small = ks_critical_value(30, 30, alpha=0.01)
    large = ks_critical_value(3000, 3000, alpha=0.01)
    assert large < small
