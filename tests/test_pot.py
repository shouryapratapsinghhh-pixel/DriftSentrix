import numpy as np
import pytest
from scipy.stats import genpareto

from anomaly.eval.pot import PotThreshold


def test_bad_level_raises():
    with pytest.raises(ValueError, match="level"):
        PotThreshold(level=1.5)


def test_bad_q_raises():
    with pytest.raises(ValueError, match="q must be"):
        PotThreshold(q=0)


def test_exceedance_rate_near_q_exponential():
    # exponential scores: a GPD with shape=0 is the exact tail model here,
    # so the fitted threshold's exceedance rate should land close to q
    rng = np.random.default_rng(0)
    scores = rng.exponential(scale=1.0, size=20000)
    pot = PotThreshold(level=0.95, q=1e-2).fit(scores)
    t = pot.threshold()

    # check exceedance rate on a large FRESH sample from the same distribution
    fresh = rng.exponential(scale=1.0, size=200000)
    empirical_rate = np.mean(fresh > t)
    assert empirical_rate == pytest.approx(1e-2, rel=0.5)  # generous tolerance, it's a tail estimate


def test_exceedance_rate_near_q_gpd_distributed():
    rng = np.random.default_rng(1)
    # true GPD tail, shape=0.2, scale=1.0
    scores = genpareto.rvs(c=0.2, scale=1.0, size=20000, random_state=rng)
    pot = PotThreshold(level=0.95, q=1e-2).fit(scores)
    t = pot.threshold()

    fresh = genpareto.rvs(c=0.2, scale=1.0, size=200000, random_state=rng)
    empirical_rate = np.mean(fresh > t)
    assert empirical_rate == pytest.approx(1e-2, rel=0.5)


def test_constant_scores_does_not_crash():
    scores = np.full(1000, 5.0)
    pot = PotThreshold(level=0.95, q=1e-3).fit(scores)
    t = pot.threshold()
    assert np.isfinite(t)


def test_few_excesses_does_not_crash():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=200)  # small n, few excesses at a high level
    pot = PotThreshold(level=0.995, q=1e-4).fit(scores)
    t = pot.threshold()
    assert np.isfinite(t)


def test_threshold_requires_fit_first():
    pot = PotThreshold()
    with pytest.raises(RuntimeError, match="fit"):
        pot.threshold()


def test_lower_q_gives_higher_threshold():
    rng = np.random.default_rng(3)
    scores = rng.exponential(scale=1.0, size=5000)
    pot = PotThreshold(level=0.95, q=1e-2).fit(scores)
    t_loose = pot.threshold(q=1e-2)
    t_strict = pot.threshold(q=1e-4)
    assert t_strict > t_loose  # rarer exceedance rate -> higher bar
