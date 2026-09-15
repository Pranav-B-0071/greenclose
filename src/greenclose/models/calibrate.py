"""Probability calibration (brief rung 7).

Raw gradient-boosting scores are not probabilities, and even a logistic head is usually overconfident.
Both calibrators are fitted on a *purged validation fold* carved out of the training window, never on
the rows the model was fitted on and never on out-of-sample rows.

The output columns are what the decision layer and the forward log consume: `prob_beat` (probability
the stock beats the day's cross-sectional mean) and `prob_green` (probability of a positive return).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    """Isotonic with a Platt (sigmoid) fallback, fitted on held-out scores."""

    method: str = "isotonic"  # isotonic | platt
    iso: IsotonicRegression | None = None
    platt: Any = None
    base_rate: float = 0.5

    def fit(self, scores: np.ndarray, outcomes: np.ndarray) -> Calibrator:
        s = np.asarray(scores, dtype="float64")
        y = np.asarray(outcomes, dtype="float64")
        ok = np.isfinite(s) & np.isfinite(y)
        s, y = s[ok], y[ok]
        self.base_rate = float(y.mean()) if len(y) else 0.5
        if len(y) < 200 or len(np.unique(y)) < 2:
            self.iso = self.platt = None
            return self
        if self.method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        else:
            self.platt = LogisticRegression(max_iter=1000).fit(s.reshape(-1, 1), y.astype(int))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype="float64")
        if self.iso is not None:
            return np.clip(np.asarray(self.iso.predict(s), dtype="float64"), 0.0, 1.0)
        if self.platt is not None:
            return np.asarray(self.platt.predict_proba(s.reshape(-1, 1))[:, 1], dtype="float64")
        return np.full(len(s), self.base_rate)


@dataclass
class CalibrationPair:
    """The two probability heads the report and the forward log need."""

    beat: Calibrator
    green: Calibrator

    @classmethod
    def fit(
        cls, scores: np.ndarray, beat: np.ndarray, green: np.ndarray, *, method: str = "isotonic"
    ) -> CalibrationPair:
        return cls(
            beat=Calibrator(method=method).fit(scores, beat),
            green=Calibrator(method=method).fit(scores, green),
        )

    def transform(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        return {"prob_beat": self.beat.transform(scores), "prob_green": self.green.transform(scores)}
