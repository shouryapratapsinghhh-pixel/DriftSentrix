import numpy as np
import pytest

from anomaly.data.loaders import make_synthetic
from anomaly.eval.metrics import pr_auc
from anomaly.models.baselines import RandomDetector
from anomaly.models.lstm_ae import LSTMAEDetector

pytestmark = pytest.mark.slow  # trains a real torch model

# small config so the whole file runs fast on CPU
TINY = {"window": 8, "hidden": 8, "layers": 1, "latent": 4, "batch_size": 32, "max_epochs": 20, "patience": 5}


@pytest.fixture(scope="module")
def synthetic_data():
    return make_synthetic(n_train=300, n_test=200, d=3, seed=0)


def test_training_loss_decreases(synthetic_data):
    train, _, _ = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0)
    det.fit(train)
    losses = det.history["train_loss"]
    assert losses[-1] < losses[0]


def test_output_shape_and_finiteness(synthetic_data):
    train, test, _ = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    scores = det.score(test)
    assert scores.shape == (len(test),)
    assert np.all(np.isfinite(scores))


def test_beats_random_on_pr_auc(synthetic_data):
    train, test, labels = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    scores = det.score(test)
    auc = pr_auc(labels, scores)

    random_aucs = [
        pr_auc(labels, RandomDetector(seed=s).fit(train).score(test)) for s in range(5)
    ]
    assert auc > np.mean(random_aucs)


def test_causality_future_perturbation_does_not_change_earlier_scores(synthetic_data):
    train, test, _ = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    scores_original = det.score(test)

    perturbed = test.copy()
    cutoff = len(test) // 2
    perturbed[cutoff:] += 100.0

    scores_perturbed = det.score(perturbed)

    # windows fully before the cutoff never see the perturbed region
    safe_end = cutoff - TINY["window"]  # last index whose window is entirely pre-cutoff
    assert safe_end > 0
    np.testing.assert_allclose(
        scores_original[:safe_end], scores_perturbed[:safe_end], rtol=1e-5, atol=1e-6
    )


def test_determinism_under_fixed_seed(synthetic_data):
    train, test, _ = synthetic_data
    det1 = LSTMAEDetector(**TINY, seed=7).fit(train)
    det2 = LSTMAEDetector(**TINY, seed=7).fit(train)
    np.testing.assert_allclose(det1.score(test), det2.score(test), rtol=1e-5)


def test_save_load_round_trip(synthetic_data, tmp_path):
    train, test, _ = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    scores_before = det.score(test)

    det.save(tmp_path / "artifact")
    loaded = LSTMAEDetector.load(tmp_path / "artifact")
    scores_after = loaded.score(test)

    np.testing.assert_allclose(scores_before, scores_after, rtol=1e-6)


def test_score_mode_last_step_differs_from_window_mean(synthetic_data):
    train, test, _ = synthetic_data
    det_mean = LSTMAEDetector(**TINY, score_mode="window_mean", seed=0).fit(train)
    det_last = LSTMAEDetector(**TINY, score_mode="last_step", seed=0).fit(train)
    # same trained weights would differ only by scoring aggregation; here
    # they're separately fit models, so just check both run and produce
    # valid, generally different score arrays (not asserting equality)
    s_mean = det_mean.score(test)
    s_last = det_last.score(test)
    assert s_mean.shape == s_last.shape
    assert not np.allclose(s_mean, s_last)


def test_explain_returns_top_k_features(synthetic_data):
    train, test, _ = synthetic_data
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    window = test[: TINY["window"]]
    result = det.explain(window, top_k=2)
    assert len(result) == 2
    assert all(isinstance(i, int) and isinstance(v, float) for i, v in result)
