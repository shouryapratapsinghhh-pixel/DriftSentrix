"""CLI: python -m anomaly.run_baselines --dataset synthetic [--root DATA_DIR --machine machine-1-1]

Fits every baseline on train, scores test, and prints PR-AUC plus
oracle-threshold point/PA/event metrics for each. This is a quick sanity
tool, not the full experiment harness (that's Phase 1's experiment.py).
"""

from __future__ import annotations

import argparse
import logging

from anomaly.data.loaders import load_nasa, load_smd, make_synthetic
from anomaly.eval.metrics import pr_auc
from anomaly.eval.thresholds import best_f1_threshold, evaluate
from anomaly.models.baselines import (
    EWMADetector,
    IsolationForestDetector,
    PCADetector,
    RandomDetector,
    ZScoreDetector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DETECTORS = [RandomDetector, ZScoreDetector, EWMADetector, IsolationForestDetector, PCADetector]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline detectors and print a results table.")
    parser.add_argument("--dataset", choices=["synthetic", "smd", "nasa"], default="synthetic")
    parser.add_argument("--root", default=None, help="data root, required for smd/nasa")
    parser.add_argument("--machine", default="machine-1-1", help="SMD machine id")
    parser.add_argument("--channel", default="A-1", help="NASA channel id")
    parser.add_argument("--spacecraft", default="SMAP")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "synthetic":
        train, test, labels = make_synthetic(seed=args.seed)
    elif args.dataset == "smd":
        if not args.root:
            parser.error("--root is required for --dataset smd")
        train, test, labels = load_smd(args.root, args.machine)
    else:
        if not args.root:
            parser.error("--root is required for --dataset nasa")
        train, test, labels = load_nasa(args.root, args.channel, args.spacecraft)

    print(f"{'model':<20}{'pr_auc':>10}{'point_f1':>12}{'pa_f1':>10}{'event_f1':>10}")
    for cls in DETECTORS:
        try:
            det = cls(seed=args.seed)
        except TypeError:
            det = cls()  # deterministic detectors (ZScore, EWMA) take no seed
        det.fit(train)
        scores = det.score(test)
        auc = pr_auc(labels, scores)
        t = best_f1_threshold(labels, scores)
        m = evaluate(labels, scores, t)
        print(
            f"{det.name:<20}{auc:>10.3f}{m['point_f1']:>12.3f}"
            f"{m['pa_f1']:>10.3f}{m['event_f1']:>10.3f}"
        )


if __name__ == "__main__":
    main()
