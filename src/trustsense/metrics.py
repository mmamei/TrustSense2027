"""Metrics that respect missing labels in ExtraSensory."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score


def _expected_calibration_error(
    truth: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if selected.any():
            error += selected.mean() * abs(truth[selected].mean() - probability[selected].mean())
    return float(error)


def _selective_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    prediction = (probability >= threshold).astype(int)
    error = (prediction != truth).astype(float)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    index_80 = max(0, int(np.ceil(0.8 * len(error))) - 1)
    return float(cumulative_risk.mean()), float(cumulative_risk[index_80])


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    confidence: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    observed = ~np.isnan(y_true)
    truth = y_true[observed].astype(int)
    prob = probability[observed]
    prediction = (prob >= threshold).astype(int)
    if len(np.unique(truth)) < 2:
        return {
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "aurc": float("nan"),
            "risk_at_80_coverage": float("nan"),
        }
    selected_confidence = (
        2.0 * np.abs(prob - 0.5) if confidence is None else confidence[observed]
    )
    aurc, risk_80 = _selective_metrics(truth, prob, selected_confidence, threshold)
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "brier": float(brier_score_loss(truth, prob)),
        "ece": _expected_calibration_error(truth, prob),
        "aurc": aurc,
        "risk_at_80_coverage": risk_80,
    }


def optimal_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select a deterministic F1 threshold using observed validation labels."""
    observed = ~np.isnan(y_true)
    truth = y_true[observed].astype(int)
    prob = probability[observed]
    if len(np.unique(truth)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199)))
    scores = np.array(
        [f1_score(truth, prob >= threshold, zero_division=0) for threshold in candidates]
    )
    best = np.flatnonzero(scores == scores.max())
    # Prefer the most conservative threshold when several yield identical F1.
    return float(candidates[best[-1]])
