"""Evaluation metrics for anomaly detection.

Three families, reported side by side (never PA-F1 alone -- see AGENTS.md
section 7):
  - point_f1: standard, strict, point-by-point precision/recall/F1.
  - pa_f1: point-adjusted -- if ANY point in a true anomalous segment is
    predicted positive, the WHOLE segment counts as detected. This is the
    metric the literature (Kim et al. 2022, Wu & Keogh 2021) shows can be
    gamed: a detector with a high enough false-positive rate can "cover"
    most segments by luck alone.
  - event_metrics: event-level precision/recall/F1 plus mean detection
    delay and false-alarm rate -- the honest operational picture.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12


def segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for each contiguous run
    of 1s in a binary label array.
    """
    labels = np.asarray(labels)
    out: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            in_seg = False
            out.append((start, i - 1))
    if in_seg:
        out.append((start, len(labels) - 1))
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def point_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Strict point-by-point precision, recall, F1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return _prf(tp, fp, fn)


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return an adjusted copy of y_pred: for each true segment, if any
    point inside it is predicted positive, mark the ENTIRE segment
    positive.
    """
    y_true = np.asarray(y_true).astype(int)
    adjusted = np.asarray(y_pred).astype(int).copy()
    for start, end in segments(y_true):
        if adjusted[start : end + 1].any():
            adjusted[start : end + 1] = 1
    return adjusted


def pa_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Point-adjusted precision, recall, F1. Report next to point_f1,
    never alone.
    """
    adjusted = point_adjust(y_true, y_pred)
    return point_f1(y_true, adjusted)


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-level precision/recall/F1 (a true segment is a 'detected
    event' if it overlaps any predicted positive point -- but unlike
    PA-F1, precision is NOT inflated: each predicted segment outside all
    true segments is one false alarm, not credited to anything), mean
    detection delay (in steps, from segment start to first true-positive
    point within it), and false alarms per 1000 steps.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    true_segs = segments(y_true)
    pred_segs = segments(y_pred)

    detected = 0
    delays: list[int] = []
    for start, end in true_segs:
        hit_idx = np.where(y_pred[start : end + 1] == 1)[0]
        if len(hit_idx) > 0:
            detected += 1
            delays.append(int(hit_idx[0]))

    # a predicted segment is a false alarm if it does not overlap any true segment
    def _overlaps_any(seg, others):
        s, e = seg
        return any(not (e < os or oe < s) for os, oe in others)

    false_alarm_segs = [seg for seg in pred_segs if not _overlaps_any(seg, true_segs)]

    n_true = len(true_segs)
    n_pred = len(pred_segs)
    event_recall = detected / n_true if n_true > 0 else 0.0
    event_precision = (n_pred - len(false_alarm_segs)) / n_pred if n_pred > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0
        else 0.0
    )
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    false_alarms_per_1000 = len(false_alarm_segs) / len(y_true) * 1000

    return {
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "mean_detection_delay": mean_delay,
        "n_false_alarms": len(false_alarm_segs),
        "false_alarms_per_1000": false_alarms_per_1000,
    }


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))
