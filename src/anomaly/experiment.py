"""Config-driven experiment runner.

python -m anomaly.experiment --config configs/synthetic_smoke.yaml

For each (entity, seed, model, threshold_mode), fits the model on train,
scores test, evaluates, and writes one row to reports/runs/<run_id>/metrics.csv
plus a copy of the resolved config as config.yaml.

Determinism: running the same config twice must yield identical metrics.csv
(same seeds -> same rows, same order). See tests/test_experiment.py.
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import time
import uuid
from pathlib import Path

import pandas as pd
import yaml

from anomaly.data.loaders import load_nasa, load_smd, make_synthetic
from anomaly.eval.pot import PotThreshold
from anomaly.eval.thresholds import best_f1_threshold, evaluate, percentile_threshold
from anomaly.models.baselines import (
    EWMADetector,
    IsolationForestDetector,
    PCADetector,
    RandomDetector,
    ZScoreDetector,
)
from anomaly.models.lstm_ae import LSTMAEDetector
from anomaly.models.transformer_ae import TransformerAEDetector

logger = logging.getLogger(__name__)

DETECTOR_REGISTRY = {
    "random": RandomDetector,
    "zscore": ZScoreDetector,
    "ewma": EWMADetector,
    "isolation_forest": IsolationForestDetector,
    "pca": PCADetector,
    "lstm_ae": LSTMAEDetector,
    "transformer_ae": TransformerAEDetector,
}

# models whose output does not depend on seed -- run once, not once per seed
DETERMINISTIC_MODELS = {"zscore", "ewma", "pca"}

VALID_DATASETS = {"synthetic", "smd", "smap", "msl"}
VALID_THRESHOLDS = {"oracle", "train_percentile", "pot"}


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    required = ["dataset", "entities", "seeds", "models", "thresholds", "output_dir"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    if config["dataset"] not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {config['dataset']!r}")

    if not isinstance(config["models"], list) or len(config["models"]) == 0:
        raise ValueError("config['models'] must be a non-empty list")
    for m in config["models"]:
        if "name" not in m:
            raise ValueError(f"model entry missing 'name': {m}")
        if m["name"] not in DETECTOR_REGISTRY:
            raise ValueError(f"unknown model name {m['name']!r}, known: {list(DETECTOR_REGISTRY)}")

    if not isinstance(config["seeds"], list) or len(config["seeds"]) == 0:
        raise ValueError("config['seeds'] must be a non-empty list")

    unknown_thresholds = set(config["thresholds"]) - VALID_THRESHOLDS
    if unknown_thresholds:
        raise ValueError(f"unknown threshold modes: {unknown_thresholds}, known: {VALID_THRESHOLDS}")

    if "train_percentile" in config["thresholds"] and not config.get("train_percentile"):
        raise ValueError(
            "threshold mode 'train_percentile' requires a non-empty "
            "config['train_percentile'] list, e.g. [99.0, 99.5]"
        )


# ---------------------------------------------------------------------------
# Entity + data resolution
# ---------------------------------------------------------------------------


def resolve_entities(config: dict) -> list[str]:
    """Turn config['entities'] (a list, or the string 'all') into a
    concrete list of entity ids for this dataset.
    """
    dataset = config["dataset"]
    entities = config["entities"]

    if dataset == "synthetic":
        return ["synthetic"]

    if entities != "all":
        return list(entities)

    root = config.get("root")
    if not root:
        raise ValueError(f"config['root'] is required to resolve entities='all' for {dataset}")

    if dataset == "smd":
        files = sorted(glob.glob(str(Path(root) / "train" / "*.txt")))
        return [Path(f).stem for f in files]

    # smap / msl
    import pandas as _pd  # local import, only needed here

    meta = _pd.read_csv(Path(root) / "labeled_anomalies.csv")
    spacecraft = "SMAP" if dataset == "smap" else "MSL"
    return meta.loc[meta["spacecraft"] == spacecraft, "chan_id"].tolist()


def load_entity_data(config: dict, entity: str):
    dataset = config["dataset"]
    if dataset == "synthetic":
        return make_synthetic(
            n_train=config.get("synthetic_n_train", 2000),
            n_test=config.get("synthetic_n_test", 1000),
            d=config.get("synthetic_d", 5),
            seed=0,  # fixed data-generation seed; model seeds vary separately
        )
    if dataset == "smd":
        return load_smd(config["root"], entity)
    spacecraft = "SMAP" if dataset == "smap" else "MSL"
    return load_nasa(config["root"], entity, spacecraft)


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------


def build_detector(name: str, params: dict, seed: int):
    cls = DETECTOR_REGISTRY[name]
    kwargs = dict(params)
    if name not in ("zscore", "ewma"):  # these two take no seed arg
        kwargs.setdefault("seed", seed)
    return cls(**kwargs)


def run_experiment(config: dict) -> pd.DataFrame:
    """Run every (entity, seed, model, threshold_mode) combination and
    return one long-form DataFrame, one row per combination.
    """
    entities = resolve_entities(config)
    rows: list[dict] = []

    for entity in entities:
        train, test, labels = load_entity_data(config, entity)
        val = train[-max(1, int(0.2 * len(train))) :]  # last 20% of train, chronological

        for model_cfg in config["models"]:
            name = model_cfg["name"]
            params = model_cfg.get("params", {})
            label = model_cfg.get("label", name)  # ablation configs disambiguate via label
            seeds = [config["seeds"][0]] if name in DETERMINISTIC_MODELS else config["seeds"]

            for seed in seeds:
                det = build_detector(name, params, seed)
                det.fit(train)
                test_scores = det.score(test)

                for mode in config["thresholds"]:
                    if mode == "oracle":
                        t = best_f1_threshold(labels, test_scores)
                        row = {"entity": entity, "seed": seed, "model": label, "threshold_mode": "oracle"}
                        row.update(evaluate(labels, test_scores, t))
                        rows.append(row)
                    elif mode == "train_percentile":
                        val_scores = det.score(val)
                        for pct in config["train_percentile"]:
                            t = percentile_threshold(val_scores, pct)
                            row = {
                                "entity": entity,
                                "seed": seed,
                                "model": label,
                                "threshold_mode": f"train_percentile_p{pct}",
                            }
                            row.update(evaluate(labels, test_scores, t))
                            rows.append(row)
                    else:  # pot
                        val_scores = det.score(val)
                        for pot_cfg in config.get("pot", [{"level": 0.98, "q": 1e-3}]):
                            pot = PotThreshold(level=pot_cfg["level"], q=pot_cfg["q"]).fit(val_scores)
                            t = pot.threshold()
                            row = {
                                "entity": entity,
                                "seed": seed,
                                "model": label,
                                "threshold_mode": f"pot_l{pot_cfg['level']}_q{pot_cfg['q']}",
                            }
                            row.update(evaluate(labels, test_scores, t))
                            rows.append(row)

    return pd.DataFrame(rows)


def write_run(config: dict, df: pd.DataFrame, run_id: str | None = None) -> Path:
    run_id = run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(config["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    logger.info("wrote run to %s", out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a config-driven experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    df = run_experiment(copy.deepcopy(config))
    write_run(config, df, run_id=args.run_id)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
