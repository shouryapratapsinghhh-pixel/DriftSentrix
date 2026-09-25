import pandas as pd
import pytest

from anomaly.report import (
    bootstrap_ci_table,
    macro_average_table,
    per_entity_table,
    seed_stats_table,
)

METRIC_COLS = [
    "point_precision",
    "point_recall",
    "point_f1",
    "pa_precision",
    "pa_recall",
    "pa_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "mean_detection_delay",
    "n_false_alarms",
    "false_alarms_per_1000",
]


def _row(entity, seed, model, mode, f1):
    """Build a minimal fake metrics row; only point_f1 is varied
    meaningfully, every other metric column is filled with the same
    value so the aggregation math is easy to hand-check.
    """
    row = {"entity": entity, "seed": seed, "model": model, "threshold_mode": mode}
    row.update({col: f1 for col in METRIC_COLS})
    return row


def test_per_entity_averages_across_seeds():
    # entity A, model x: seeds 0,1,2 with point_f1 = 0.2, 0.4, 0.6 -> mean 0.4
    df = pd.DataFrame(
        [
            _row("A", 0, "x", "oracle", 0.2),
            _row("A", 1, "x", "oracle", 0.4),
            _row("A", 2, "x", "oracle", 0.6),
        ]
    )
    out = per_entity_table(df)
    assert len(out) == 1
    assert out.iloc[0]["point_f1"] == pytest.approx(0.4)


def test_macro_average_weights_entities_equally():
    # entity A has 1 row (f1=1.0), entity B has 3 rows averaging to 0.0
    # macro-average must be (1.0 + 0.0) / 2 = 0.5, NOT weighted by row count
    df = pd.DataFrame(
        [
            _row("A", 0, "x", "oracle", 1.0),
            _row("B", 0, "x", "oracle", 0.0),
            _row("B", 1, "x", "oracle", 0.0),
            _row("B", 2, "x", "oracle", 0.0),
        ]
    )
    per_entity = per_entity_table(df)
    macro = macro_average_table(per_entity)
    assert len(macro) == 1
    assert macro.iloc[0]["point_f1"] == 0.5


def test_seed_stats_std_zero_for_single_seed():
    df = pd.DataFrame([_row("A", 0, "zscore", "oracle", 0.9)])
    out = seed_stats_table(df)
    assert out.iloc[0]["point_f1_std"] == 0.0
    assert out.iloc[0]["point_f1_mean"] == 0.9


def test_seed_stats_nonzero_std_across_seeds():
    df = pd.DataFrame(
        [
            _row("A", 0, "iforest", "oracle", 0.2),
            _row("A", 1, "iforest", "oracle", 0.8),
        ]
    )
    out = seed_stats_table(df)
    assert out.iloc[0]["point_f1_std"] > 0


def test_bootstrap_ci_degenerate_with_one_entity():
    # one entity -> every bootstrap resample is that same entity ->
    # CI collapses to a point (lo == hi == the true value)
    df = pd.DataFrame([_row("A", 0, "x", "oracle", 0.7)])
    per_entity = per_entity_table(df)
    ci = bootstrap_ci_table(per_entity, n_resamples=200, seed=0)
    assert ci.iloc[0]["point_f1_ci_lo"] == 0.7
    assert ci.iloc[0]["point_f1_ci_hi"] == 0.7


def test_bootstrap_ci_contains_true_mean_with_multiple_entities():
    df = pd.DataFrame(
        [_row(f"E{i}", 0, "x", "oracle", v) for i, v in enumerate([0.2, 0.4, 0.6, 0.8, 1.0])]
    )
    per_entity = per_entity_table(df)
    macro = macro_average_table(per_entity)
    true_mean = macro.iloc[0]["point_f1"]

    ci = bootstrap_ci_table(per_entity, n_resamples=2000, seed=0)
    assert ci.iloc[0]["point_f1_ci_lo"] <= true_mean <= ci.iloc[0]["point_f1_ci_hi"]


def test_bootstrap_ci_is_seeded_reproducible():
    df = pd.DataFrame(
        [_row(f"E{i}", 0, "x", "oracle", v) for i, v in enumerate([0.1, 0.5, 0.9])]
    )
    per_entity = per_entity_table(df)
    ci1 = bootstrap_ci_table(per_entity, n_resamples=500, seed=42)
    ci2 = bootstrap_ci_table(per_entity, n_resamples=500, seed=42)
    pd.testing.assert_frame_equal(ci1, ci2)
