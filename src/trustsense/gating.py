"""Learned modality-integrity gating from validation-time fault augmentation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def probability_tensor(
    probability_by_key: dict[tuple[str, str], np.ndarray],
    modalities: list[str],
    labels: list[str],
) -> np.ndarray:
    """Return samples by modalities by labels."""
    return np.stack(
        [
            np.column_stack(
                [probability_by_key[(modality, label)] for label in labels]
            )
            for modality in modalities
        ],
        axis=1,
    )


def gate_features(probabilities: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Build one integrity feature vector per sample and modality."""
    n_samples, n_modalities, _ = probabilities.shape
    disagreement = np.empty((n_samples, n_modalities), dtype=float)
    for modality in range(n_modalities):
        others = np.delete(probabilities, modality, axis=1)
        consensus = np.median(others, axis=1)
        disagreement[:, modality] = np.mean(
            np.abs(probabilities[:, modality, :] - consensus), axis=1
        )

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    entropy = -np.mean(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped),
        axis=2,
    )
    margin = np.mean(2.0 * np.abs(probabilities - 0.5), axis=2)
    modality_one_hot = np.broadcast_to(
        np.eye(n_modalities)[None, :, :], (n_samples, n_modalities, n_modalities)
    )
    numeric = np.stack([base, margin, entropy, disagreement], axis=2)
    return np.concatenate([numeric, modality_one_hot], axis=2)


def fit_integrity_gate(
    clean_probabilities: np.ndarray,
    clean_base: np.ndarray,
    corrupted_states: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> HistGradientBoostingClassifier:
    """Fit on clean modalities and the corrupted target from each scenario."""
    clean_features = gate_features(clean_probabilities, clean_base)
    x_parts = [clean_features.reshape(-1, clean_features.shape[-1])]
    y_parts = [np.ones(clean_features.shape[0] * clean_features.shape[1], dtype=int)]

    for target_index, probabilities, base in corrupted_states:
        features = gate_features(probabilities, base)
        x_parts.append(features[:, target_index, :])
        y_parts.append(np.zeros(features.shape[0], dtype=int))

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    counts = np.bincount(y, minlength=2)
    sample_weight = np.where(y == 0, len(y) / (2 * counts[0]), len(y) / (2 * counts[1]))
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def integrity_probability(
    model: HistGradientBoostingClassifier,
    probabilities: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    features = gate_features(probabilities, base)
    flat = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
    return flat.reshape(features.shape[:2]) * (base > 0)


def gated_fusion(
    probabilities: np.ndarray,
    integrity: np.ndarray,
    *,
    minimum_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse every label using a shared per-modality integrity estimate."""
    available = (integrity > 0).astype(float)
    weights = available * (minimum_weight + (1.0 - minimum_weight) * integrity)
    denominator = weights.sum(axis=1)
    fallback = probabilities.mean(axis=1)
    fused = np.divide(
        (probabilities * weights[:, :, None]).sum(axis=1),
        denominator[:, None],
        out=fallback.copy(),
        where=denominator[:, None] > 0,
    )
    return fused, denominator

