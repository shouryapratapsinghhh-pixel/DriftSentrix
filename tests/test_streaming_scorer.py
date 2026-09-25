import numpy as np
import pytest

from anomaly.data.loaders import make_synthetic
from anomaly.models.baselines import (
    EWMADetector,
    IsolationForestDetector,
    PCADetector,
    ZScoreDetector,
)
from anomaly.models.lstm_ae import LSTMAEDetector
from anomaly.models.transformer_ae import TransformerAEDetector
from anomaly.streaming.scorer import StreamingScorer

pytestmark = pytest.mark.slow  # trains a real torch model

ALL_DETECTOR_FACTORIES = {
    "zscore": lambda: ZScoreDetector(),
    "ewma": lambda: EWMADetector(alpha=0.2),
    "isolation_forest": lambda: IsolationForestDetector(seed=0),
    "pca": lambda: PCADetector(seed=0),
    "lstm_ae": lambda: LSTMAEDetector(
        window=8, hidden=8, layers=1, latent=4, batch_size=32, max_epochs=10, patience=3, seed=0
    ),
    "transformer_ae": lambda: TransformerAEDetector(
        window=8, d_model=16, nhead=2, layers=1, dim_feedforward=32,
        batch_size=32, max_epochs=10, patience=3, seed=0,
    ),
}


@pytest.fixture(scope="module")
def synthetic_data():
    return make_synthetic(n_train=300, n_test=100, d=3, seed=0)


@pytest.mark.parametrize("name", list(ALL_DETECTOR_FACTORIES))
def test_streaming_equals_batch(name, synthetic_data):
    """THE critical test: pushing points one at a time through
    StreamingScorer must produce exactly the same scores as batch
    score(), from index window-1 onward (before that, streaming is
    still in warmup and legitimately returns no score).
    """
    train, test, _ = synthetic_data
    det = ALL_DETECTOR_FACTORIES[name]()
    det.fit(train)

    batch_scores = det.score(test)

    scorer = StreamingScorer(det, threshold=1e18)  # threshold irrelevant here, only checking scores
    streaming_scores = []
    warmup_flags = []
    for row in test:
        result = scorer.push(row)
        streaming_scores.append(result.score)
        warmup_flags.append(result.warmup)

    window = getattr(det, "window", 1)
    # before the window fills, streaming must report warmup (no score)
    assert all(warmup_flags[: window - 1])
    assert not any(warmup_flags[window - 1 :])

    streaming_scores = np.array(streaming_scores[window - 1 :], dtype=np.float64)
    batch_from_window = np.array(batch_scores[window - 1 :], dtype=np.float64)
    np.testing.assert_allclose(streaming_scores, batch_from_window, rtol=1e-4, atol=1e-6)


def test_alert_flag_respects_threshold(synthetic_data):
    train, test, _ = synthetic_data
    det = ZScoreDetector().fit(train)
    scorer = StreamingScorer(det, threshold=0.5)
    saw_alert = False
    saw_no_alert = False
    for row in test:
        result = scorer.push(row)
        if result.alert:
            saw_alert = True
        else:
            saw_no_alert = True
    assert saw_alert and saw_no_alert  # threshold actually discriminates on this data


def test_reset_clears_state(synthetic_data):
    train, test, _ = synthetic_data
    det = ZScoreDetector().fit(train)
    scorer = StreamingScorer(det, threshold=1e18)
    for row in test[:5]:
        scorer.push(row)
    scorer.reset()
    # after reset, warmup should behave exactly as a fresh scorer would
    fresh = StreamingScorer(ZScoreDetector().fit(train), threshold=1e18)
    r_reset = scorer.push(test[0])
    r_fresh = fresh.push(test[0])
    assert r_reset.warmup == r_fresh.warmup
    assert r_reset.score == r_fresh.score


def test_per_entity_isolation(synthetic_data):
    """Two StreamingScorer instances (representing two entities) must
    not share state.
    """
    train, test, _ = synthetic_data
    det_a = EWMADetector(alpha=0.2).fit(train)
    det_b = EWMADetector(alpha=0.2).fit(train)
    scorer_a = StreamingScorer(det_a, threshold=1e18)
    scorer_b = StreamingScorer(det_b, threshold=1e18)

    for row in test[:10]:
        scorer_a.push(row)
    # b hasn't seen anything yet -- its first push should equal a fresh
    # detector's very first score, NOT reflect a's history
    r_b_first = scorer_b.push(test[0])
    fresh_det = EWMADetector(alpha=0.2).fit(train)
    fresh_scorer = StreamingScorer(fresh_det, threshold=1e18)
    r_fresh = fresh_scorer.push(test[0])
    assert r_b_first.score == pytest.approx(r_fresh.score)
