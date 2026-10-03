"""Post-hoc calibration fitted on participant-disjoint validation data."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Map arbitrary real-valued decision scores to calibrated probabilities."""

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("Platt calibration requires both classes")
        self._model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
        self._model.fit(scores, labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        return self._model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]

