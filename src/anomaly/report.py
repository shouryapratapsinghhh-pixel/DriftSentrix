"""Aggregate a run's long-form metrics.csv into honest summary tables.

python -m anomaly.report --run-id <id>      # or omit for the most recent run

Four tables, in increasing order of what they average over:
  per_entity      -- mean across seeds, per (entity, model, threshold_mode)
  macro_average   -- per_entity averaged again across entities
                      (equal weight per entity, not per row -- an entity
                      with more seeds/rows doesn't dominate)
  seed_stats      -- mean +/- std across seeds, entity dropped (model,
                      threshold_mode only) -- "how much does this model's
                      score vary run to run"
  bootstrap_ci    -- 95% CI on the macro-average, built by resampling
                      ENTITIES (not rows) with replacement, 1000 times,
                      seeded. With one entity (synthetic) this is
                      degenerate by construction: every resample is that
                      one entity, so the CI collapses to a point. That's
                      correct, not a bug -- it becomes informative once
                      SMD's 8+ machines are in the mix.

Writes reports/results.md and reports/results.csv (the per_entity table,
the one closest to a "results" section).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

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

GROUP_KEYS = ["entity", "model", "threshold_mode"]


def per_entity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean across seeds, one row per (entity, model, threshold_mode)."""
    return df.groupby(GROUP_KEYS, as_index=False)[METRIC_COLS].mean()


def macro_average_table(per_entity: pd.DataFrame) -> pd.DataFrame:
    """per_entity averaged again across entities -- equal weight per
    entity regardless of how many seeds/rows fed into its mean.
    """
    return per_entity.groupby(["model", "threshold_mode"], as_index=False)[METRIC_COLS].mean()


def seed_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds, entity dropped -- variance across runs
    of the SAME model, not across entities.
    """
    grouped = df.groupby(["model", "threshold_mode"])[METRIC_COLS]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)  # single-seed groups (deterministic models) -> std 0
    mean.columns = [f"{c}_mean" for c in mean.columns]
    std.columns = [f"{c}_std" for c in std.columns]
    return pd.concat([mean, std], axis=1).reset_index()


def bootstrap_ci_table(
    per_entity: pd.DataFrame, n_resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """95% percentile bootstrap CI on the macro-average, resampling
    entities (with replacement) rather than rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (model, mode), group in per_entity.groupby(["model", "threshold_mode"]):
        entities = group["entity"].to_numpy()
        values = group[METRIC_COLS].to_numpy()
        n = len(entities)

        boot_means = np.empty((n_resamples, len(METRIC_COLS)))
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)  # resample entity indices with replacement
            boot_means[i] = values[idx].mean(axis=0)

        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
        row = {"model": model, "threshold_mode": mode}
        for col, lo_v, hi_v in zip(METRIC_COLS, lo, hi):
            row[f"{col}_ci_lo"] = lo_v
            row[f"{col}_ci_hi"] = hi_v
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_entity = per_entity_table(df)
    return {
        "per_entity": per_entity,
        "macro_average": macro_average_table(per_entity),
        "seed_stats": seed_stats_table(df),
        "bootstrap_ci": bootstrap_ci_table(per_entity),
    }


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ablation setting (mean +/- std over seeds).

    This is seed_stats_table under a clearer name for the ablation-sweep
    use case: run_experiment writes each config entry's distinct
    window/hidden/score_mode combination into the `model` column via its
    `label` field, so grouping by (model, threshold_mode) here IS
    grouping by ablation setting.
    """
    return seed_stats_table(df)


def _find_latest_run(runs_dir: str = "reports/runs") -> str:
    candidates = sorted(glob.glob(os.path.join(runs_dir, "*")), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"no runs found under {runs_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a run's metrics.csv into report tables.")
    parser.add_argument("--run-id", default=None, help="reports/runs/<run-id>; default: most recent")
    parser.add_argument("--runs-dir", default="reports/runs")
    args = parser.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run_id) if args.run_id else _find_latest_run(args.runs_dir)
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))

    tables = report(df)

    os.makedirs("reports", exist_ok=True)
    tables["per_entity"].to_csv("reports/results.csv", index=False)

    with open("reports/results.md", "w") as f:
        f.write(f"# Results (run: {run_dir})\n\n")
        for name, table in tables.items():
            f.write(f"## {name}\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n")

        pa_inflation_path = "reports/pa_inflation.csv"
        if os.path.exists(pa_inflation_path):
            pa_df = pd.read_csv(pa_inflation_path)
            f.write("## pa_inflation_study\n\n")
            f.write(
                "Model PA-F1 vs. a random detector predicting positive at the SAME rate. "
                "A small gap means PA-F1 was mostly rewarding flagging enough points, not "
                "finding the right ones.\n\n"
            )
            f.write(pa_df.to_markdown(index=False))
            f.write("\n\n")

    print(f"aggregated {run_dir} -> reports/results.md, reports/results.csv")
    print(tables["macro_average"][["model", "threshold_mode", "point_f1", "pa_f1", "event_f1"]])


if __name__ == "__main__":
    main()
