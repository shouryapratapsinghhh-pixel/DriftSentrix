"""Baseline detectors.

Contract, shared by every detector in this repo (deep models later
implement the same interface): fit(train) -> self; score(x) -> ndarray
of shape (len(x),), higher = more anomalous, one score per timestep.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

EPS = 1e-8


class RandomDetector:
    """Sanity-check baseline: pure noise. If a real model can't beat
    this on PR-AUC, something is wrong.
    """

    name = "random"
    window = 1

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, train: np.ndarray) -> RandomDetector:
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.rng.random(len(x))


class ZScoreDetector:
    """Per-timestep max absolute z-score across features, using
    mean/std fit on train only.
    """

    name = "zscore"
    window = 1

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> ZScoreDetector:
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0) + EPS
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.abs((x - self.mean_) / self.std_)
        return z.max(axis=1)


class EWMADetector:
    """Score = deviation from an exponentially-weighted moving average
    of the stream itself (recurrent state, causal by construction).
    alpha controls how fast the EWMA adapts (higher = faster).
    """

    name = "ewma"
    window = 1

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None  # train mean, used to init EWMA
        self.std_: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> EWMADetector:
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0) + EPS
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        ewma = self.mean_.copy()
        out = np.empty(len(x))
        for i, row in enumerate(x):
            out[i] = np.abs((row - ewma) / self.std_).max()
            ewma = self.alpha * row + (1 - self.alpha) * ewma
        return out

    # -- streaming: EWMA's score depends on ALL history, not just a fixed
    # window, so it needs genuine recurrent state -- a ring-buffer replay
    # of the last `window` points (window=1 here) would silently reset
    # the EWMA on every push and diverge from the batch result. See
    # streaming/scorer.py: it uses these methods when present instead of
    # the generic windowed path.

    def reset_stream(self) -> None:
        self._stream_ewma = self.mean_.copy()

    def update_stream(self, x_t: np.ndarray) -> float:
        if not hasattr(self, "_stream_ewma"):
            self.reset_stream()
        score = float(np.abs((x_t - self._stream_ewma) / self.std_).max())
        self._stream_ewma = self.alpha * x_t + (1 - self.alpha) * self._stream_ewma
        return score


class IsolationForestDetector:
    name = "isolation_forest"
    window = 1

    def __init__(self, seed: int = 0, n_estimators: int = 100):
        self.model = IsolationForest(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        )

    def fit(self, train: np.ndarray) -> IsolationForestDetector:
        self.model.fit(train)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal, so negate
        return -self.model.decision_function(x)


class PCADetector:
    """Reconstruction error under a low-rank PCA fit on train."""

    name = "pca"
    window = 1

    def __init__(self, n_components: float = 0.9, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.model: PCA | None = None

    def fit(self, train: np.ndarray) -> PCADetector:
        self.model = PCA(n_components=self.n_components, random_state=self.seed)
        self.model.fit(train)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        reconstructed = self.model.inverse_transform(self.model.transform(x))
        return np.mean((x - reconstructed) ** 2, axis=1)
