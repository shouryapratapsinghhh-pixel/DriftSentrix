"""Application state: loads a servable artifact (see export.py) once at
startup, holds one StreamingScorer per entity_id (each with its own,
isolated buffer -- see streaming/scorer.py), enforces a max-entities cap
with TTL eviction, and tracks request/alert counters + latency
percentiles for /metrics.

NOTE on sharing the underlying detector across entities: this is safe
for the deep models this service currently loads (lstm_ae,
transformer_ae) because their score() is a pure function of (fixed
trained weights, input window) -- no per-call mutable state on the
detector itself. It would NOT be safe for EWMADetector, whose
update_stream() stores recurrent state ON THE DETECTOR OBJECT (see
models/baselines.py) -- sharing one EWMADetector across entities would
let them corrupt each other's state. export.py only offers lstm_ae /
transformer_ae for exactly this reason; if EWMA-over-HTTP is ever
wanted, each entity would need its OWN detector instance, not a shared
one.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from anomaly.models.lstm_ae import LSTMAEDetector
from anomaly.models.transformer_ae import TransformerAEDetector
from anomaly.monitor.drift import DriftMonitor
from anomaly.streaming.scorer import StreamingScorer

DETECTOR_CLASSES = {"lstm_ae": LSTMAEDetector, "transformer_ae": TransformerAEDetector}


class _EntityEntry:
    __slots__ = ("drift_monitor", "last_used", "scorer")

    def __init__(self, scorer: StreamingScorer, drift_monitor: DriftMonitor | None):
        self.scorer = scorer
        self.drift_monitor = drift_monitor
        self.last_used = time.time()


class AppState:
    def __init__(
        self,
        detector,
        detector_type: str,
        threshold: float,
        threshold_mode: str,
        n_features: int,
        reference: np.ndarray | None = None,
        max_entities: int = 1000,
        ttl_seconds: float = 3600.0,
        drift_window: int = 500,
    ):
        self.detector = detector
        self.detector_type = detector_type
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.n_features = n_features
        self.reference = reference  # raw (unscaled) train subsample; None -> drift monitoring disabled
        self.max_entities = max_entities
        self.ttl_seconds = ttl_seconds
        self.drift_window = drift_window

        self._entities: dict[str, _EntityEntry] = {}

        self.requests_total = 0
        self.alerts_total = 0
        self._latencies_ms: deque = deque(maxlen=5000)

    # -- per-entity scorer + drift-monitor registry --------------------------

    def _get_entry(self, entity_id: str) -> _EntityEntry:
        self._evict_expired()
        entry = self._entities.get(entity_id)
        if entry is None:
            if len(self._entities) >= self.max_entities:
                raise RuntimeError(f"max entities ({self.max_entities}) reached")
            drift_monitor = (
                DriftMonitor(self.reference, window=self.drift_window) if self.reference is not None else None
            )
            entry = _EntityEntry(StreamingScorer(self.detector, self.threshold), drift_monitor)
            self._entities[entity_id] = entry
        entry.last_used = time.time()
        return entry

    def get_scorer(self, entity_id: str) -> StreamingScorer:
        return self._get_entry(entity_id).scorer

    def get_drift_monitor(self, entity_id: str) -> DriftMonitor | None:
        """Does NOT create a new entity entry if one doesn't already exist
        -- checking drift for an entity that's never pushed anything makes
        no sense, and creating one here would let /drift/{id} silently
        consume an entity-cap slot as a side effect of a read.
        """
        self._evict_expired()
        entry = self._entities.get(entity_id)
        return entry.drift_monitor if entry else None

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._entities.items() if now - v.last_used > self.ttl_seconds]
        for k in expired:
            del self._entities[k]

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    # -- metrics -----------------------------------------------------------

    def record_request(self, latency_ms: float, alert: bool = False) -> None:
        self.requests_total += 1
        if alert:
            self.alerts_total += 1
        self._latencies_ms.append(latency_ms)

    def latency_percentiles(self) -> tuple[float, float, float]:
        if not self._latencies_ms:
            return (0.0, 0.0, 0.0)
        arr = np.array(self._latencies_ms)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        return float(p50), float(p95), float(p99)


def load_state(model_dir: str | Path) -> AppState:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    with open(model_dir / "threshold.json") as f:
        threshold_info = json.load(f)

    detector_type = config["detector_type"]
    if detector_type not in DETECTOR_CLASSES:
        raise ValueError(f"unknown detector_type {detector_type!r} in {model_dir / 'config.json'}")

    detector = DETECTOR_CLASSES[detector_type].load(model_dir)
    n_features = len(detector.mean_)

    reference_path = model_dir / "reference.npz"
    reference = np.load(reference_path)["reference"] if reference_path.exists() else None

    return AppState(
        detector=detector,
        detector_type=detector_type,
        threshold=threshold_info["value"],
        threshold_mode=threshold_info["mode"],
        n_features=n_features,
        reference=reference,
    )
