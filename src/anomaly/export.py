"""Export a trained detector + a deployable (label-free) threshold to a
servable artifact directory.

python -m anomaly.export --config configs/synthetic_smoke.yaml --model lstm_ae \
    --threshold-mode pot --pot-level 0.98 --pot-q 0.001 --out artifacts/model_v1

Only train_percentile and pot are offered here -- NOT oracle, since oracle
needs test labels that a real deployment doesn't have. (If you want an
oracle-threshold artifact for local experimentation, build one by hand;
this CLI deliberately doesn't make that easy to reach for by accident.)

Writes, under --out:
  model.pt        -- torch weights (from the detector's own save())
  config.json      -- detector hyperparameters + "detector_type" (from save(),
                       with detector_type added so serve/app.py knows which
                       class to reconstruct)
  scaler.npz       -- train mean/std (from save())
  threshold.json   -- {"value": ..., "mode": ..., "params": {...}}
  reference.npz    -- a subsample of raw (unscaled) train data, used as the
                       drift monitor's reference distribution (see
                       monitor/drift.py, serve/state.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly.eval.pot import PotThreshold
from anomaly.eval.thresholds import percentile_threshold
from anomaly.experiment import build_detector, load_config, load_entity_data

MAX_REFERENCE_ROWS = 2000  # cap so reference.npz stays small regardless of train size


def export_artifact(
    config: dict,
    model_name: str,
    threshold_mode: str,
    out_dir: str | Path,
    entity: str = "synthetic",
    percentile: float = 99.5,
    pot_level: float = 0.98,
    pot_q: float = 1e-3,
) -> Path:
    if threshold_mode not in ("train_percentile", "pot"):
        raise ValueError(
            f"threshold_mode must be 'train_percentile' or 'pot' (not oracle -- "
            f"a deployment doesn't have test labels), got {threshold_mode!r}"
        )

    model_cfg = next((m for m in config["models"] if m["name"] == model_name), None)
    if model_cfg is None:
        raise ValueError(f"no model named {model_name!r} in config['models']")

    train, _test, _labels = load_entity_data(config, entity)
    val = train[-max(1, int(0.2 * len(train))) :]

    seed = config["seeds"][0]
    det = build_detector(model_name, model_cfg.get("params", {}), seed)
    det.fit(train)

    val_scores = det.score(val)
    if threshold_mode == "train_percentile":
        t = percentile_threshold(val_scores, percentile)
        threshold_info = {"mode": "train_percentile", "value": t, "params": {"percentile": percentile}}
    else:
        pot = PotThreshold(level=pot_level, q=pot_q).fit(val_scores)
        t = pot.threshold()
        threshold_info = {"mode": "pot", "value": t, "params": {"level": pot_level, "q": pot_q}}

    out_dir = Path(out_dir)
    det.save(out_dir)  # writes model.pt, config.json, scaler.npz

    config_path = out_dir / "config.json"
    with open(config_path) as f:
        det_config = json.load(f)
    det_config["detector_type"] = model_name
    with open(config_path, "w") as f:
        json.dump(det_config, f)

    with open(out_dir / "threshold.json", "w") as f:
        json.dump(threshold_info, f)

    rng = np.random.default_rng(seed)
    n_ref = min(MAX_REFERENCE_ROWS, len(train))
    ref_idx = rng.choice(len(train), size=n_ref, replace=False)
    np.savez(out_dir / "reference.npz", reference=train[ref_idx])

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained detector to a servable artifact.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["lstm_ae", "transformer_ae"])
    parser.add_argument("--entity", default="synthetic")
    parser.add_argument("--threshold-mode", required=True, choices=["train_percentile", "pot"])
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--pot-level", type=float, default=0.98)
    parser.add_argument("--pot-q", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = export_artifact(
        config,
        args.model,
        args.threshold_mode,
        args.out,
        entity=args.entity,
        percentile=args.percentile,
        pot_level=args.pot_level,
        pot_q=args.pot_q,
    )
    print(f"exported artifact to {out_dir}")


if __name__ == "__main__":
    main()
