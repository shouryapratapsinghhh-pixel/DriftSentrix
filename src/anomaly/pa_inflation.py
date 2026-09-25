"""PA-inflation study.

For a model's predictions at some threshold, build a RANDOM detector
that predicts positive at the SAME rate (same fraction of flagged
points), then compare PA-F1 for both. If the random, rate-matched
detector's PA-F1 is close to the real model's, PA-F1 was mostly
rewarding "flag enough points somewhere" rather than actually finding
the right ones -- the headline finding this whole repo is built to show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.eval.metrics import pa_f1, point_f1


def pa_inflation_row(
    entity: str, model: str, labels: np.ndarray, y_pred: np.ndarray, seed: int = 0
) -> dict:
    labels = np.asarray(labels)
    y_pred = np.asarray(y_pred)
    positive_rate = float(y_pred.mean())

    rng = np.random.default_rng(seed)
    random_pred = (rng.random(len(labels)) < positive_rate).astype(int)

    _, _, model_point = point_f1(labels, y_pred)
    _, _, model_pa = pa_f1(labels, y_pred)
    _, _, random_point = point_f1(labels, random_pred)
    _, _, random_pa = pa_f1(labels, random_pred)

    return {
        "entity": entity,
        "model": model,
        "positive_rate": positive_rate,
        "model_point_f1": model_point,
        "model_pa_f1": model_pa,
        "random_point_f1": random_point,
        "random_pa_f1": random_pa,
        "pa_inflation": model_pa - model_point,  # how much PA-F1 flatters THIS model
        "random_pa_inflation": random_pa - random_point,  # how much it flatters pure noise
    }


def pa_inflation_study(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
