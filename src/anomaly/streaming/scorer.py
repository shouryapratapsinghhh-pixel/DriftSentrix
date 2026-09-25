"""Streaming scorer: turns any Detector into a causal, stateful, one-
timestep-at-a-time scorer.

Two paths, chosen automatically per detector:

1. Detectors with update_stream()/reset_stream() (currently: EWMA --
   its score depends on ALL history, not a fixed window, so it needs
   genuine recurrent state). Score is available immediately, no warmup.

2. Everything else: a preallocated ring buffer holding the last
   `detector.window` points. No score until the buffer is full (warmup);
   once full, each push re-scores the current window. Because the
   window is causal (the score-per-window convention pads earlier
   points, but scoring a buffer of EXACTLY `window` length always
   produces one window and hence one score, taken directly, no
   padding involved) this is mathematically identical to what
   detector.score() would produce for a window ending at this index in
   batch mode -- see the streaming==batch equivalence test, the most
   important test in this file's test suite.

Per-entity state is isolated simply by using one StreamingScorer
instance per entity -- no shared mutable state between instances.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreResult:
    score: float | None
    alert: bool
    warmup: bool


class StreamingScorer:
    def __init__(self, detector, threshold: float):
        self.detector = detector
        self.threshold = threshold
        self._stateful = hasattr(detector, "update_stream")
        if self._stateful:
            detector.reset_stream()
        else:
            window = getattr(detector, "window", 1)
            self._buffer: deque = deque(maxlen=window)
            self._window = window

    def reset(self) -> None:
        if self._stateful:
            self.detector.reset_stream()
        else:
            self._buffer.clear()

    def push(self, x_t: np.ndarray) -> ScoreResult:
        x_t = np.asarray(x_t)

        if self._stateful:
            score = self.detector.update_stream(x_t)
            return ScoreResult(score=score, alert=score >= self.threshold, warmup=False)

        self._buffer.append(x_t)
        if len(self._buffer) < self._window:
            return ScoreResult(score=None, alert=False, warmup=True)

        window_array = np.stack(self._buffer, axis=0)  # (window, d)
        scores = self.detector.score(window_array)  # length == window, all identical (see module docstring)
        score = float(scores[-1])
        return ScoreResult(score=score, alert=score >= self.threshold, warmup=False)
