"""Regression gate: run a fixed, seeded synthetic mini-benchmark and fail
if PR-AUC or event-F1 drops by more than `tolerance` versus a committed
baseline (reports/baseline_metrics.json).

python scripts/regression_gate.py                  # check against the committed baseline
python scripts/regression_gate.py --update         # refresh the baseline (review the diff before committing!)
python scripts/regression_gate.py --tolerance 0.05  # looser tolerance

Deliberately synthetic and cheap (baselines only by default; deep models
are slow and their PR-AUC has run-to-run seed variance beyond what a
tolerance-based gate should absorb -- pass --include-deep to add them
locally, but CI runs baselines-only for speed and determinism).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from anomaly.data.loaders import make_synthetic
from anomaly.eval.metrics import event_metrics, pr_auc
from anomaly.eval.thresholds import best_f1_threshold
from anomaly.experiment import build_detector

DEFAULT_BASELINE_PATH = "reports/baseline_metrics.json"
DEFAULT_TOLERANCE = 0.03
GATE_SEED = 0
BASELINE_MODELS = ["random", "zscore", "ewma", "isolation_forest", "pca"]
DEEP_MODELS = ["lstm_ae", "transformer_ae"]


def compute_metrics(model_name: str, seed: int = GATE_SEED) -> dict:
    """Fit+score one model on a fixed synthetic mini-benchmark, return
    its PR-AUC and (oracle-threshold) event-F1.
    """
    train, test, labels = make_synthetic(n_train=1000, n_test=500, d=4, seed=GATE_SEED)
    if model_name in DEEP_MODELS:
        kwargs = {
            "window": 8, "max_epochs": 10, "patience": 3, "batch_size": 32, "seed": seed,
        }
        if model_name == "lstm_ae":
            kwargs.update(hidden=8, layers=1, latent=4)
        else:
            kwargs.update(d_model=16, nhead=2, layers=1, dim_feedforward=32)
        det = build_detector(model_name, kwargs, seed)
    else:
        det = build_detector(model_name, {}, seed)
    det.fit(train)
    scores = det.score(test)

    auc = pr_auc(labels, scores)
    t = best_f1_threshold(labels, scores)
    y_pred = (scores >= t).astype(int)
    ev = event_metrics(labels, y_pred)
    return {"pr_auc": auc, "event_f1": ev["event_f1"]}


def compute_all_metrics(models: list[str]) -> dict:
    return {m: compute_metrics(m) for m in models}


def evaluate_gate(current: dict, baseline: dict, tolerance: float) -> tuple[bool, list[str]]:
    """Pure comparison logic, no model fitting -- kept separate from
    compute_all_metrics so this (the part with actual decision logic) is
    fast and cheap to unit-test.
    """
    passed = True
    lines = []
    for model, metrics in current.items():
        baseline_for_model = baseline.get(model, {})
        for metric_name, cur_val in metrics.items():
            base_val = baseline_for_model.get(metric_name)
            if base_val is None:
                lines.append(f"{model:20s} {metric_name:10s} no baseline entry -- skipped")
                continue
            drop = base_val - cur_val
            status = "REGRESSION" if drop > tolerance else "OK"
            if status == "REGRESSION":
                passed = False
            lines.append(
                f"{model:20s} {metric_name:10s} baseline={base_val:.4f} current={cur_val:.4f} "
                f"drop={drop:+.4f} tolerance={tolerance:.4f} [{status}]"
            )
    return passed, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression gate against a committed metrics baseline.")
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--models", nargs="+", default=BASELINE_MODELS)
    parser.add_argument("--include-deep", action="store_true", help="also gate lstm_ae/transformer_ae")
    parser.add_argument("--update", action="store_true", help="write current metrics as the new baseline")
    args = parser.parse_args()

    models = list(args.models)
    if args.include_deep:
        models += DEEP_MODELS

    current = compute_all_metrics(models)

    if args.update:
        os.makedirs(os.path.dirname(args.baseline_path) or ".", exist_ok=True)
        with open(args.baseline_path, "w") as f:
            json.dump(current, f, indent=2)
        print(f"baseline written to {args.baseline_path} -- review the diff before committing")
        return

    if not os.path.exists(args.baseline_path):
        print(f"no baseline at {args.baseline_path} -- run with --update to create one. Nothing to gate.")
        sys.exit(1)

    with open(args.baseline_path) as f:
        baseline = json.load(f)

    passed, lines = evaluate_gate(current, baseline, args.tolerance)
    print("\n".join(lines))

    if not passed:
        print("\nREGRESSION GATE FAILED")
        sys.exit(1)
    print("\nregression gate passed")


if __name__ == "__main__":
    main()
