"""Measure StreamingScorer.push() latency (p50/p95/p99) and throughput,
per model. Writes reports/latency.json.

python scripts/bench_latency.py --model lstm_ae --n-pushes 2000
python scripts/bench_latency.py --model lstm_ae --http --host http://127.0.0.1:8000

torch.set_num_threads(1) is used for stable, comparable numbers --
multi-threaded BLAS/torch ops make latency benchmarks noisy and
machine-dependent otherwise.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from anomaly.data.loaders import make_synthetic
from anomaly.experiment import DETECTOR_REGISTRY
from anomaly.streaming.scorer import StreamingScorer

TINY = {"window": 16, "hidden": 16, "layers": 1, "latent": 8, "batch_size": 32, "max_epochs": 5, "patience": 2}


def _percentiles(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99)}


def bench_in_process(model_name: str, n_pushes: int, n_warmup: int, seed: int = 0) -> dict:
    torch.set_num_threads(1)
    train, _test, _labels = make_synthetic(n_train=500, d=5, seed=seed)

    if model_name in ("lstm_ae", "transformer_ae"):
        kwargs = dict(TINY)
        if model_name == "transformer_ae":
            kwargs = {
                "window": 16, "d_model": 16, "nhead": 2, "layers": 1, "dim_feedforward": 32,
                "batch_size": 32, "max_epochs": 5, "patience": 2,
            }
        det = DETECTOR_REGISTRY[model_name](**kwargs, seed=seed)
    else:
        det = DETECTOR_REGISTRY[model_name]()
    det.fit(train)

    scorer = StreamingScorer(det, threshold=1e18)  # threshold irrelevant to timing
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_pushes + n_warmup, train.shape[1])).astype(np.float32)

    for i in range(n_warmup):  # warm up (JIT/cache effects, first-call overhead)
        scorer.push(points[i])

    latencies_ms = []
    start_all = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_pushes):
        t0 = time.perf_counter()
        scorer.push(points[i])
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total_s = time.perf_counter() - start_all

    result = {
        "model": model_name,
        "mode": "in_process",
        "n_pushes": n_pushes,
        "throughput_points_per_s": n_pushes / total_s,
        **_percentiles(latencies_ms),
    }
    return result


def bench_http(model_name: str, host: str, n_pushes: int, n_warmup: int, seed: int = 0) -> dict:
    import httpx

    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_pushes + n_warmup, 3)).tolist()  # assumes a 3-feature exported model

    with httpx.Client(base_url=host, timeout=10.0) as client:
        for i in range(n_warmup):
            client.post("/stream/bench_entity/push", json={"x": points[i]})

        latencies_ms = []
        start_all = time.perf_counter()
        for i in range(n_warmup, n_warmup + n_pushes):
            t0 = time.perf_counter()
            client.post("/stream/bench_entity/push", json={"x": points[i]})
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        total_s = time.perf_counter() - start_all

    return {
        "model": model_name,
        "mode": "http",
        "n_pushes": n_pushes,
        "throughput_points_per_s": n_pushes / total_s,
        **_percentiles(latencies_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark streaming-scorer push latency/throughput.")
    parser.add_argument("--model", default="zscore", choices=list(DETECTOR_REGISTRY))
    parser.add_argument("--n-pushes", type=int, default=1000)
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--http", action="store_true", help="benchmark a running server instead")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.http:
        result = bench_http(args.model, args.host, args.n_pushes, args.n_warmup, args.seed)
    else:
        result = bench_in_process(args.model, args.n_pushes, args.n_warmup, args.seed)

    print(json.dumps(result, indent=2))
    out_path = f"reports/latency_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
