"""Fast, reproducible probabilistic baselines."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_sgd(
    *, seed: int, alpha: float, max_iter: int, tolerance: float
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=tolerance,
                    random_state=seed,
                    average=True,
                ),
            ),
        ]
    )

