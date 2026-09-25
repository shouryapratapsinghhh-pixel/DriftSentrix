"""Peaks-Over-Threshold (POT) thresholding.

Deployable (no labels needed): fit on train/validation scores alone.
Idea: pick an initial "high" empirical threshold t (the `level`
quantile), look at how far the scores that exceed it overshoot t (the
"excesses"), and fit a Generalized Pareto Distribution (GPD) to those
excesses -- extreme value theory says GPD is the right asymptotic model
for tail excesses over a high threshold, regardless of the underlying
distribution. Then extrapolate to a threshold whose expected exceedance
rate is exactly `q` (a small tail probability, e.g. 1e-3), even into
the tail beyond what was actually observed.

Caveat (see docs/EVALUATION.md): fitted on train scores, so it can be
miscalibrated if the test-time score distribution has shifted.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import genpareto

logger = logging.getLogger(__name__)

MIN_EXCESSES_FOR_FIT = 10  # below this, a GPD fit is unreliable; warn and fall back


class PotThreshold:
    def __init__(self, level: float = 0.98, q: float = 1e-3):
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")
        self.level = level
        self.q = q

        self.t_: float | None = None
        self.shape_: float | None = None
        self.scale_: float | None = None
        self.n_: int | None = None
        self.n_exceed_: int | None = None

    def fit(self, train_scores: np.ndarray) -> PotThreshold:
        train_scores = np.asarray(train_scores, dtype=np.float64)
        self.n_ = len(train_scores)
        self.t_ = float(np.quantile(train_scores, self.level))

        excesses = train_scores[train_scores > self.t_] - self.t_
        self.n_exceed_ = len(excesses)

        if self.n_exceed_ == 0:
            logger.warning(
                "POT: no excesses over the level-%.3f threshold (constant or degenerate "
                "scores?); falling back to the empirical threshold itself.",
                self.level,
            )
            self.shape_, self.scale_ = 0.0, 0.0
            return self

        if self.n_exceed_ < MIN_EXCESSES_FOR_FIT:
            logger.warning(
                "POT: only %d excesses (< %d) -- GPD fit is unreliable at this level.",
                self.n_exceed_,
                MIN_EXCESSES_FOR_FIT,
            )

        if np.allclose(excesses, excesses[0]):
            # degenerate: all excesses identical -> genpareto.fit can misbehave
            self.shape_, self.scale_ = 0.0, max(float(excesses[0]), 1e-8)
            return self

        try:
            shape, _loc, scale = genpareto.fit(excesses, floc=0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                raise ValueError("non-finite or non-positive scale from genpareto.fit")
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see method-of-moments fallback
            logger.warning("POT: genpareto.fit failed (%s), using method-of-moments fallback.", e)
            mean = excesses.mean()
            var = excesses.var()
            if var <= 0 or mean <= 0:
                shape, scale = 0.0, max(mean, 1e-8)
            else:
                shape = 0.5 * (mean**2 / var - 1)
                scale = 0.5 * mean * (mean**2 / var + 1)

        self.shape_, self.scale_ = float(shape), float(scale)
        return self

    def threshold(self, q: float | None = None) -> float:
        """The threshold whose expected exceedance rate (over the ORIGINAL
        score distribution, not just the tail) is `q`.
        """
        if self.t_ is None:
            raise RuntimeError("call fit() before threshold()")
        q = self.q if q is None else q

        if self.n_exceed_ == 0:
            return self.t_

        ratio = q * self.n_ / self.n_exceed_
        xi, sigma = self.shape_, self.scale_

        if abs(xi) < 1e-6:
            return self.t_ - sigma * np.log(ratio)
        return self.t_ + (sigma / xi) * (ratio ** (-xi) - 1)
