import numpy as np
import pytest

from anomaly.data.loaders import make_synthetic
from anomaly.eval.metrics import pr_auc
from anomaly.models.baselines import (
    EWMADetector,
    IsolationForestDetector,
    PCADetector,
    RandomDetector,
    ZScoreDetector,
)

ALL_DETECTORS = [RandomDetector, ZScoreDetector, EWMADetector, IsolationForestDetector, PCADetector]


@pytest.fixture(scope="module")
def synthetic_data():
    return make_synthetic(n_train=500, n_test=300, d=4, seed=0)


@pytest.mark.parametrize("cls", ALL_DETECTORS)
def test_shape_and_finiteness(cls, synthetic_data):
    train, test, _ = synthetic_data
    det = cls()
    det.fit(train)
    scores = det.score(test)
    assert scores.shape == (len(test),)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("cls", [ZScoreDetector, EWMADetector, IsolationForestDetector, PCADetector])
def test_beats_random_on_synthetic(cls, synthetic_data):
    train, test, labels = synthetic_data
    det = cls()
    det.fit(train)
    scores = det.score(test)
    auc = pr_auc(labels, scores)

    random_aucs = []
    for seed in range(5):
        r = RandomDetector(seed=seed).fit(train)
        random_aucs.append(pr_auc(labels, r.score(test)))
    assert auc > np.mean(random_aucs)


@pytest.mark.parametrize("cls", [ZScoreDetector, EWMADetector, IsolationForestDetector, PCADetector])
def test_causality_future_perturbation_does_not_change_earlier_scores(cls, synthetic_data):
    train, test, _ = synthetic_data
    det = cls()
    det.fit(train)
    scores_original = det.score(test)

    perturbed = test.copy()
    cutoff = len(test) // 2
    perturbed[cutoff:] += 100.0  # blow up everything after the cutoff

    det2 = cls()
    det2.fit(train)
    scores_perturbed = det2.score(perturbed)

    np.testing.assert_allclose(scores_original[:cutoff], scores_perturbed[:cutoff], rtol=1e-5)


def test_random_detector_is_seeded_deterministic():
    train, test, _ = make_synthetic(seed=1)
    a = RandomDetector(seed=42).fit(train).score(test)
    b = RandomDetector(seed=42).fit(train).score(test)
    np.testing.assert_array_equal(a, b)
