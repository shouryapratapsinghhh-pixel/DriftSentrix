"""Dataset loaders.

Contract: every loader returns (train, test, labels) where
    train:  (n_train, d) float32, no anomalies (assumed clean)
    test:   (n_test, d) float32
    labels: (n_test,) int, 1 = anomalous, 0 = normal

make_synthetic needs no external data, so tests and CI run fully offline.
load_smd / load_nasa read real data from `root` and raise a clear
FileNotFoundError if it isn't there -- they are not required by tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_synthetic(
    n_train: int = 2000,
    n_test: int = 1000,
    d: int = 5,
    anomaly_rate: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic multivariate stream: sinusoids + noise, with
    injected mean/variance-shift anomalies in the test set only.
    """
    rng = np.random.default_rng(seed)

    def _clean(n: int) -> np.ndarray:
        t = np.arange(n)
        x = np.stack(
            [np.sin(2 * np.pi * t / (50 + 10 * i) + i) for i in range(d)], axis=1
        )
        x += rng.normal(scale=0.1, size=x.shape)
        return x.astype(np.float32)

    train = _clean(n_train)

    test = _clean(n_test)
    labels = np.zeros(n_test, dtype=int)

    n_anomalies = max(1, int(n_test * anomaly_rate))
    # place a few contiguous anomalous segments, not scattered single points
    n_segments = max(1, n_anomalies // 20)
    seg_len = max(5, n_anomalies // n_segments)
    starts = rng.choice(n_test - seg_len - 1, size=n_segments, replace=False)
    for s in starts:
        e = s + seg_len
        test[s:e] += rng.normal(loc=3.0, scale=1.0, size=(e - s, d))
        labels[s:e] = 1

    return train, test.astype(np.float32), labels


def load_smd(root: str | Path, machine: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one Server Machine Dataset entity, e.g. machine='machine-1-1'.

    Expects root/train/<machine>.txt, root/test/<machine>.txt,
    root/test_label/<machine>.txt (comma-separated values, one row per
    timestep; label file is one 0/1 per line) -- the standard SMD layout.
    """
    root = Path(root)
    train_path = root / "train" / f"{machine}.txt"
    test_path = root / "test" / f"{machine}.txt"
    label_path = root / "test_label" / f"{machine}.txt"
    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"SMD file not found: {p}. See data/README.md for download instructions."
            )
    train = pd.read_csv(train_path, header=None).to_numpy(copy=True).astype(np.float32)
    test = pd.read_csv(test_path, header=None).to_numpy(copy=True).astype(np.float32)
    labels = pd.read_csv(label_path, header=None).to_numpy(copy=True).astype(int).ravel()
    return train, test, labels


def load_nasa(
    root: str | Path, channel: str, spacecraft: str = "SMAP"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one NASA SMAP/MSL channel.

    Expects root/train/<channel>.npy, root/test/<channel>.npy, and
    root/labeled_anomalies.csv with columns including chan_id, spacecraft,
    anomaly_sequences (a string-encoded list of [start, end] pairs,
    ASSUMED INCLUSIVE -- verify with scripts/inspect_labels.py).
    """
    root = Path(root)
    train_path = root / "train" / f"{channel}.npy"
    test_path = root / "test" / f"{channel}.npy"
    label_csv = root / "labeled_anomalies.csv"
    for p in (train_path, test_path, label_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"NASA file not found: {p}. See data/README.md for download instructions."
            )
    train = np.load(train_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)

    meta = pd.read_csv(label_csv)
    row = meta[(meta["chan_id"] == channel) & (meta["spacecraft"] == spacecraft)]
    if row.empty:
        raise ValueError(f"No label metadata for channel={channel} spacecraft={spacecraft}")

    import ast

    sequences = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
    labels = np.zeros(len(test), dtype=int)
    for start, end in sequences:
        labels[start : end + 1] = 1  # inclusive -- TODO(owner): confirm via inspect_labels.py

    return train, test, labels
