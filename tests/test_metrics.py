import numpy as np

from anomaly.eval.metrics import event_metrics, pa_f1, point_adjust, point_f1, pr_auc, segments


def test_segments_basic():
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert segments(labels) == [(1, 2), (5, 5), (7, 9)]


def test_segments_trailing_run():
    labels = np.array([0, 0, 1, 1])
    assert segments(labels) == [(2, 3)]


def test_segments_none():
    assert segments(np.zeros(5, dtype=int)) == []


def test_point_f1_hand_computed():
    # true: positions 2,3 anomalous. pred: positions 3,4 flagged.
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 1])
    p, r, f1 = point_f1(y_true, y_pred)
    # tp=1 (pos 3), fp=1 (pos 4), fn=1 (pos 2)
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_point_f1_perfect():
    y = np.array([0, 1, 1, 0])
    p, r, f1 = point_f1(y, y)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_point_f1_no_predictions():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 0, 0])
    p, r, f1 = point_f1(y_true, y_pred)
    assert p == 0.0  # no positive predictions -> precision defined as 0
    assert r == 0.0
    assert f1 == 0.0


def test_point_adjust_fills_segment():
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])  # single hit inside the segment
    adjusted = point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(adjusted, [0, 1, 1, 1, 0])


def test_point_adjust_no_hit_no_fill():
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    adjusted = point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(adjusted, [0, 0, 0, 0, 0])


def test_pa_f1_inflation_demo():
    """This is the headline finding: PA-F1 can look much better than
    point-F1 for a detector that just happens to land one hit per segment.
    """
    y_true = np.array([0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # 1 hit/segment
    _, _, f1_point = point_f1(y_true, y_pred)
    _, _, f1_pa = pa_f1(y_true, y_pred)
    assert f1_pa > f1_point  # PA-F1 inflates here
    assert f1_pa == 1.0  # both segments fully "detected" from one point each


def test_event_metrics_delay_and_false_alarms():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    # detects the segment 2 steps late (hits at index 4), plus one false alarm segment at 7-8
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 0])
    m = event_metrics(y_true, y_pred)
    assert m["event_recall"] == 1.0
    assert m["n_false_alarms"] == 1
    assert m["mean_detection_delay"] == 2  # hit at offset 2 within the true segment (idx 2..4)


def test_event_metrics_missed_segment():
    y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0])  # first segment caught, second missed
    m = event_metrics(y_true, y_pred)
    assert m["event_recall"] == 0.5
    assert m["n_false_alarms"] == 0


def test_pr_auc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert pr_auc(y_true, scores) == 1.0


def test_pr_auc_no_positives_returns_nan():
    y_true = np.zeros(5, dtype=int)
    scores = np.random.rand(5)
    assert np.isnan(pr_auc(y_true, scores))
