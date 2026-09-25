"""python -m anomaly.scripts.pa_inflation_study --config configs/synthetic_smoke.yaml

For each (entity, model, seed) in a config, fits/scores the model, takes
its oracle-threshold predictions, runs the PA-inflation comparison
against a rate-matched random detector, and writes:
  reports/pa_inflation.csv
  reports/figures/pa_inflation.png
"""

from __future__ import annotations

import argparse
import copy
import os

from anomaly.eval.thresholds import best_f1_threshold
from anomaly.experiment import (
    DETERMINISTIC_MODELS,
    build_detector,
    load_config,
    load_entity_data,
    resolve_entities,
)
from anomaly.pa_inflation import pa_inflation_row
from anomaly.viz import plot_pa_inflation


def run(config: dict) -> pd.DataFrame:  # noqa: F821 -- pandas imported lazily below
    import pandas as pd

    entities = resolve_entities(config)
    rows = []
    for entity in entities:
        train, test, labels = load_entity_data(config, entity)
        for model_cfg in config["models"]:
            name = model_cfg["name"]
            params = model_cfg.get("params", {})
            label = model_cfg.get("label", name)
            seeds = [config["seeds"][0]] if name in DETERMINISTIC_MODELS else config["seeds"]
            for seed in seeds:
                det = build_detector(name, params, seed)
                det.fit(train)
                scores = det.score(test)
                t = best_f1_threshold(labels, scores)  # oracle threshold for this study
                y_pred = (scores >= t).astype(int)
                rows.append(pa_inflation_row(entity, label, labels, y_pred, seed=seed))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PA-inflation study.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    df = run(copy.deepcopy(config))

    os.makedirs("reports/figures", exist_ok=True)
    df.to_csv("reports/pa_inflation.csv", index=False)
    plot_pa_inflation(df, "reports/figures/pa_inflation.png")

    print(df.to_string(index=False))
    print("\nwrote reports/pa_inflation.csv, reports/figures/pa_inflation.png")


if __name__ == "__main__":
    main()
