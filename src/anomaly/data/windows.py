"""Sliding-window utilities.

Convention: a window's score is assigned to its LAST timestep (causal --
by the time the window closes, you know the score). In batch mode, the
first `window - 1` timesteps (which never complete a window) get the
first window's score, so the output array always has one score per
input timestep.
"""

from __future__ import annotations

import numpy as np


def make_windows(x: np.ndarray, window: int) -> np.ndarray:
    """(n, d) -> (n - window + 1, window, d) sliding windows, stride 1."""
    n = x.shape[0]
    if window > n:
        raise ValueError(f"window ({window}) larger than series length ({n})")
    d = x.shape[1]
    # as_strided view, no copy, then copy out for safety
    stride0, stride1 = x.strides
    shape = (n - window + 1, window, d)
    strides = (stride0, stride0, stride1)
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return windows.copy()


def window_scores_to_points(window_scores: np.ndarray, window: int, n_points: int) -> np.ndarray:
    """Map one score per window -> one score per original timestep.

    len(window_scores) must equal n_points - window + 1. The score for
    window i is assigned to timestep (window - 1 + i). Timesteps
    [0, window - 2] (before the first window closes) get the first
    window's score.
    """
    expected = n_points - window + 1
    if len(window_scores) != expected:
        raise ValueError(f"expected {expected} window scores, got {len(window_scores)}")
    out = np.empty(n_points, dtype=window_scores.dtype)
    out[window - 1 :] = window_scores
    out[: window - 1] = window_scores[0]
    return out
