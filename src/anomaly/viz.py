"""Visualizations -- headless (Agg backend), always savefig, never plt.show().

(a) plot_timeline       -- top features + anomaly score + threshold + true
                            events shaded + predicted alerts, for one entity.
(b) plot_metric_bars    -- grouped bar chart per model: point-F1 vs PA-F1
                            vs event-F1.
(c) plot_pa_inflation   -- the headline-finding figure: a model's PA-F1
                            next to a RANDOM detector's PA-F1 at the SAME
                            predicted-positive rate. If random gets close
                            to the real model, PA-F1 was inflating.
(d) plot_detection_delay_boxplot -- delay distribution per model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless -- must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly.eval.metrics import segments


def plot_timeline(
    test: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    out_path: str,
    feature_indices: list[int] | None = None,
    feature_names: list[str] | None = None,
) -> str:
    """test: (n, d). Top panel: a few raw feature channels. Bottom panel:
    anomaly score with threshold line, true events shaded, predicted
    alerts marked.
    """
    n, d = test.shape
    if feature_indices is None:
        # "top features" = highest-variance channels, a reasonable default
        # when no specific features are requested
        feature_indices = list(np.argsort(-test.var(axis=0))[: min(3, d)])
    names = feature_names or [f"feature_{i}" for i in range(d)]

    t = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for i in feature_indices:
        ax1.plot(t, test[:, i], label=names[i], linewidth=1)
    ax1.set_ylabel("raw value")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Top features")

    ax2.plot(t, scores, color="black", linewidth=1, label="anomaly score")
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1, label="threshold")
    for start, end in segments(labels):
        ax2.axvspan(start, end, color="orange", alpha=0.3)
    alerts = np.where(scores >= threshold)[0]
    ax2.scatter(alerts, scores[alerts], color="red", s=8, zorder=5, label="predicted alert")
    ax2.set_ylabel("score")
    ax2.set_xlabel("timestep")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("Anomaly score (orange = true event, red = predicted alert)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_metric_bars(
    df: pd.DataFrame,
    out_path: str,
    metrics: tuple[str, ...] = ("point_f1", "pa_f1", "event_f1"),
    model_col: str = "model",
) -> str:
    """df: one row per model (already filtered to a single threshold_mode
    -- pass a pre-aggregated table, e.g. macro_average_table at oracle).
    """
    models = df[model_col].tolist()
    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Point-F1 vs PA-F1 vs Event-F1 by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_pa_inflation(df: pd.DataFrame, out_path: str, model_col: str = "model") -> str:
    """df: output of pa_inflation.pa_inflation_study -- one row per
    (entity, model[, seed]) with model_pa_f1 and random_pa_f1 columns.
    Aggregates to one bar-pair per model.
    """
    agg = df.groupby(model_col)[["model_pa_f1", "random_pa_f1"]].mean().reset_index()
    models = agg[model_col].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    ax.bar(x - width / 2, agg["model_pa_f1"], width, label="model PA-F1", color="steelblue")
    ax.bar(
        x + width / 2, agg["random_pa_f1"], width,
        label="random (rate-matched) PA-F1", color="lightcoral",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("PA-F1")
    ax.set_title("PA-inflation study: real model vs. rate-matched random detector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_detection_delay_boxplot(
    df: pd.DataFrame, out_path: str, model_col: str = "model", delay_col: str = "mean_detection_delay"
) -> str:
    """df: long-form, one row per (entity, seed, model, ...) with a
    per-run detection-delay value -- boxplot shows the SPREAD across
    entities/seeds per model, not just a single mean.
    """
    models = sorted(df[model_col].unique())
    data = [df.loc[df[model_col] == m, delay_col].dropna().to_numpy() for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    ax.boxplot(data, tick_labels=models)
    ax.set_ylabel("mean detection delay (steps)")
    ax.set_title("Detection delay by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
