"""Availability- and reliability-aware late fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReliabilityProfile:
    median: pd.Series
    iqr: pd.Series
    expected_completeness: float
    ood_threshold: float
    ood_scale: float


def _robust_distance(values: pd.DataFrame, median: pd.Series, iqr: pd.Series) -> np.ndarray:
    scale = iqr.replace(0, 1.0).fillna(1.0)
    standardized = (values - median).abs().divide(scale)
    # Median aggregation keeps one extreme feature from dominating a modality.
    return standardized.median(axis=1, skipna=True).fillna(np.inf).to_numpy(dtype=float)


def fit_reliability_profile(train: pd.DataFrame) -> ReliabilityProfile:
    median = train.median(axis=0, skipna=True)
    q1 = train.quantile(0.25)
    q3 = train.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    train_completeness = train.notna().mean(axis=1)
    present = train_completeness > 0
    expected_completeness = float(train_completeness[present].median())

    train_distance = _robust_distance(train, median, iqr)
    finite = train_distance[np.isfinite(train_distance)]
    if len(finite) == 0:
        raise ValueError("Training data contain no finite reliability distances")
    threshold = float(np.quantile(finite, 0.95))
    tail = finite[finite >= threshold]
    scale = float(np.quantile(tail - threshold, 0.75)) if len(tail) else 1.0
    scale = max(scale, 0.25)
    return ReliabilityProfile(median, iqr, expected_completeness, threshold, scale)


def reliability_components(
    values: pd.DataFrame, profile: ReliabilityProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate completeness and robust OOD reliability components."""
    completeness = values.notna().mean(axis=1).to_numpy(dtype=float)
    relative_completeness = np.clip(
        completeness / max(profile.expected_completeness, 1e-6), 0.0, 1.0
    )
    distance = _robust_distance(values, profile.median, profile.iqr)
    excess = np.maximum(distance - profile.ood_threshold, 0.0)
    ood = np.exp(-np.minimum(excess / profile.ood_scale, 50.0))
    ood[~np.isfinite(distance)] = 0.0
    return relative_completeness, ood


def base_reliability(values: pd.DataFrame, profile: ReliabilityProfile) -> np.ndarray:
    completeness, ood = reliability_components(values, profile)
    return completeness * ood


def uniform_fusion(probabilities: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Average available modalities; reliability is rows by modalities."""
    available = (reliability > 0).astype(float)
    denominator = available.sum(axis=1)
    fallback = probabilities.mean(axis=1)
    return np.divide(
        (probabilities * available).sum(axis=1),
        denominator,
        out=fallback.copy(),
        where=denominator > 0,
    )


def reliability_fusion(
    probabilities: np.ndarray,
    base: np.ndarray,
    *,
    disagreement_strength: float = 4.0,
    reliability_floor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse one label and return probabilities plus total evidence weight."""
    n_modalities = probabilities.shape[1]
    consensus = np.empty_like(probabilities)
    for modality in range(n_modalities):
        others = np.delete(probabilities, modality, axis=1)
        other_base = np.delete(base, modality, axis=1)
        denominator = other_base.sum(axis=1)
        fallback = np.median(others, axis=1)
        consensus[:, modality] = np.divide(
            (others * other_base).sum(axis=1),
            denominator,
            out=fallback,
            where=denominator > 0,
        )

    if not 0.0 <= reliability_floor <= 1.0:
        raise ValueError("reliability_floor must lie in [0, 1]")
    agreement = np.exp(-disagreement_strength * np.abs(probabilities - consensus))
    available = (base > 0).astype(float)
    # The floor prevents normal cross-modal diversity from being mistaken for a
    # fault. A completely missing modality still receives exactly zero weight.
    weights = available * (
        reliability_floor + (1.0 - reliability_floor) * base * agreement
    )
    denominator = weights.sum(axis=1)
    fallback = uniform_fusion(probabilities, base)
    fused = np.divide(
        (probabilities * weights).sum(axis=1),
        denominator,
        out=fallback.copy(),
        where=denominator > 0,
    )
    return fused, denominator
