"""Threshold selection.

best_f1_threshold is the ORACLE: it uses test labels directly, so it's
an upper bound, never a deployable choice -- report it labeled as such.
percentile_threshold is deployable: computed from scores alone (in
practice, from train/validation scores), no labels needed.
"""

from __future__ import annotations

import numpy as np

from anomaly.eval.metrics import event_metrics, pa_f1, point_f1


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ORACLE threshold: search candidate thresholds (unique score
    values) and return the one maximizing point-F1 against y_true.
    Uses test labels -- not a deployable choice, upper-bound only.
    """
    candidates = np.unique(scores)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        y_pred = (scores >= t).astype(int)
        _, _, f1 = point_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    """Deployable threshold: the given percentile of a score distribution
    (pass train or validation scores, not test).
    """
    return float(np.percentile(scores, percentile))


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute point, point-adjusted, and event metrics at one threshold."""
    y_pred = (scores >= threshold).astype(int)
    p, r, f1 = point_f1(y_true, y_pred)
    pa_p, pa_r, pa_f1_val = pa_f1(y_true, y_pred)
    ev = event_metrics(y_true, y_pred)
    return {
        "threshold": threshold,
        "point_precision": p,
        "point_recall": r,
        "point_f1": f1,
        "pa_precision": pa_p,
        "pa_recall": pa_r,
        "pa_f1": pa_f1_val,
        **ev,
    }
