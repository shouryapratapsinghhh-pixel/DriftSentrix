"""Drift monitoring: is the live data still distributed the way the
training data was? Two independent, from-scratch signals per feature:

- PSI (Population Stability Index): buckets the reference distribution
  into quantile bins, compares how the current window's proportions in
  those same bins have shifted. 0.1/0.25 are the conventional
  warn/alert cutoffs -- INDUSTRY RULES OF THUMB, not derived from any
  statistical guarantee. Treat them as configurable defaults, not truth.

  PSI is also sample-size sensitive: with the default 10 bins, a
  current window of only ~30-50 points can show a "warn"-level PSI on
  pure sampling noise, even drawn from the exact same distribution as
  the reference (verified directly: see the module's own manual check
  during development -- a 50-point i.i.d. draw from the reference
  showed PSI ~0.19-0.23 on every feature; a 300-point draw of the same
  data showed PSI ~0.01-0.02). MIN_CURRENT_FOR_REPORT=30 is a crash
  guard, not a "this is statistically trustworthy" guarantee --
  anything relying on PSI for real decisions should look at several
  hundred current points per report, not the bare minimum.

- KS statistic (two-sample Kolmogorov-Smirnov): the max gap between the
  reference and current empirical CDFs, implemented here via sort +
  searchsorted (verified against scipy.stats.ks_2samp in tests). Turned
  into an alert using the classic ASYMPTOTIC critical value for a given
  significance level alpha -- c(alpha) = sqrt(-0.5 * ln(alpha/2)), then
  D_critical = c(alpha) * sqrt((n1+n2)/(n1*n2)) -- rather than calling
  scipy for a p-value at decision time. This keeps the whole alerting
  path self-contained; scipy is used only in tests, to confirm this
  repo's ks_statistic() matches its ks_2samp() D-value.

DriftMonitor treats a 1D reference/current the same as a single-column
2D one, so passing a 1D array of anomaly SCORES (not raw features) as
`reference` gives score-distribution drift monitoring for free, with no
separate code path.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

MIN_CURRENT_FOR_REPORT = 30  # below this, a report would be too noisy to trust


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # reference is numerically constant -- PSI's quantile-binning breaks down;
        # fall back to a direct "did anything move at all" check
        moved = not np.allclose(current, edges[0])
        return 10.0 if moved else 0.0  # 10.0: a large sentinel, well past the default alert cutoff

    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf  # outer bins catch current values more extreme than any seen in reference

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_props = np.clip(ref_counts / len(reference), eps, None)
    cur_props = np.clip(cur_counts / len(current), eps, None)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS D-statistic: max absolute gap between the two
    empirical CDFs, evaluated at every distinct value in either sample.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    all_values = np.union1d(a, b)  # sorted union of distinct values from both samples
    cdf_a = np.searchsorted(a, all_values, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_critical_value(n1: int, n2: int, alpha: float) -> float:
    c = math.sqrt(-0.5 * math.log(alpha / 2))
    return c * math.sqrt((n1 + n2) / (n1 * n2))


class DriftMonitor:
    def __init__(
        self,
        reference: np.ndarray,
        window: int = 500,
        psi_warn: float = 0.1,
        psi_alert: float = 0.25,
        ks_alpha: float = 0.01,
        feature_names: list[str] | None = None,
    ):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)  # 1D input (e.g. anomaly scores) -> single "feature"
        self.reference = reference
        self.n_features = reference.shape[1]
        self.window = window
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.ks_alpha = ks_alpha
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]
        self._buffer: deque = deque(maxlen=window)

    def update(self, x_t: np.ndarray | float) -> None:
        x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
        self._buffer.append(x_t)

    def report(self) -> dict:
        if len(self._buffer) < MIN_CURRENT_FOR_REPORT:
            return {
                "status": "insufficient_data",
                "n_current": len(self._buffer),
                "n_required": MIN_CURRENT_FOR_REPORT,
                "features": [],
                "top_drifted_features": [],
            }

        current = np.stack(self._buffer, axis=0)  # (n_current, n_features)
        features = []
        for i in range(self.n_features):
            ref_col = self.reference[:, i]
            cur_col = current[:, i]

            if np.allclose(ref_col, ref_col[0]) and np.allclose(cur_col, cur_col[0]) and ref_col[0] == cur_col[0]:
                # both constant and identical -- genuinely no drift, skip the (undefined) KS critical value
                features.append(
                    {"name": self.feature_names[i], "psi": 0.0, "ks": 0.0, "ks_alert": False, "psi_alert": False}
                )
                continue

            psi_val = psi(ref_col, cur_col)
            ks_val = ks_statistic(ref_col, cur_col)
            ks_crit = ks_critical_value(len(ref_col), len(cur_col), self.ks_alpha)
            features.append(
                {
                    "name": self.feature_names[i],
                    "psi": psi_val,
                    "ks": ks_val,
                    "ks_alert": ks_val > ks_crit,
                    "psi_alert": psi_val >= self.psi_alert,
                }
            )

        any_alert = any(f["psi"] >= self.psi_alert or f["ks_alert"] for f in features)
        any_warn = any(f["psi"] >= self.psi_warn for f in features)
        status = "alert" if any_alert else ("warn" if any_warn else "ok")

        top_drifted = sorted(features, key=lambda f: f["psi"], reverse=True)[:3]

        return {
            "status": status,
            "n_current": len(self._buffer),
            "n_required": MIN_CURRENT_FOR_REPORT,
            "features": features,
            "top_drifted_features": [f["name"] for f in top_drifted],
        }
