import os

import numpy as np
import pandas as pd
import pytest

from anomaly.pa_inflation import pa_inflation_row, pa_inflation_study
from anomaly.viz import (
    plot_detection_delay_boxplot,
    plot_metric_bars,
    plot_pa_inflation,
    plot_timeline,
)


def _non_empty_png(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


# -- viz: headless PNGs, non-empty -----------------------------------------


def test_plot_timeline_writes_nonempty_png(tmp_path):
    rng = np.random.default_rng(0)
    test = rng.normal(size=(200, 4))
    labels = np.zeros(200, dtype=int)
    labels[50:60] = 1
    scores = rng.random(200)
    out = str(tmp_path / "timeline.png")
    plot_timeline(test, labels, scores, threshold=0.5, out_path=out)
    assert _non_empty_png(out)


def test_plot_metric_bars_writes_nonempty_png(tmp_path):
    df = pd.DataFrame(
        {"model": ["zscore", "pca"], "point_f1": [0.6, 0.7], "pa_f1": [0.7, 0.8], "event_f1": [0.65, 0.75]}
    )
    out = str(tmp_path / "bars.png")
    plot_metric_bars(df, out)
    assert _non_empty_png(out)


def test_plot_pa_inflation_writes_nonempty_png(tmp_path):
    df = pd.DataFrame(
        {
            "entity": ["e1", "e1"],
            "model": ["zscore", "pca"],
            "model_pa_f1": [0.8, 0.9],
            "random_pa_f1": [0.3, 0.2],
        }
    )
    out = str(tmp_path / "pa_inflation.png")
    plot_pa_inflation(df, out)
    assert _non_empty_png(out)


def test_plot_detection_delay_boxplot_writes_nonempty_png(tmp_path):
    df = pd.DataFrame(
        {
            "model": ["zscore", "zscore", "pca", "pca"],
            "mean_detection_delay": [1.0, 2.0, 0.5, 1.5],
        }
    )
    out = str(tmp_path / "delay.png")
    plot_detection_delay_boxplot(df, out)
    assert _non_empty_png(out)


def test_no_plot_calls_show(monkeypatch, tmp_path):
    """Nothing should try to open an interactive window -- if plt.show()
    were ever called, this would fail/hang outside a headless CI runner.
    """
    import matplotlib.pyplot as plt

    def _fail_if_called(*a, **k):
        raise AssertionError("plt.show() must never be called -- headless only")

    monkeypatch.setattr(plt, "show", _fail_if_called)

    df = pd.DataFrame({"model": ["zscore"], "point_f1": [0.5], "pa_f1": [0.6], "event_f1": [0.55]})
    plot_metric_bars(df, str(tmp_path / "x.png"))  # must not raise


# -- pa_inflation: the actual math ------------------------------------------


def test_pa_inflation_row_perfect_predictor_has_zero_gap():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    row = pa_inflation_row("e1", "perfect", labels, labels.copy(), seed=0)
    assert row["model_point_f1"] == 1.0
    assert row["model_pa_f1"] == 1.0
    assert row["pa_inflation"] == 0.0


def test_pa_inflation_row_positive_rate_matches():
    labels = np.array([0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0])  # rate = 0.4
    row = pa_inflation_row("e1", "m", labels, y_pred, seed=0)
    assert row["positive_rate"] == pytest.approx(0.4)


def test_pa_inflation_row_seeded_reproducible():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    r1 = pa_inflation_row("e1", "m", labels, y_pred, seed=42)
    r2 = pa_inflation_row("e1", "m", labels, y_pred, seed=42)
    assert r1["random_pa_f1"] == r2["random_pa_f1"]


def test_pa_inflation_study_builds_dataframe():
    rows = [
        pa_inflation_row("e1", "zscore", np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), seed=0),
        pa_inflation_row("e1", "pca", np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]), seed=0),
    ]
    df = pa_inflation_study(rows)
    assert len(df) == 2
    assert set(df["model"]) == {"zscore", "pca"}
