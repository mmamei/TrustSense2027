"""Controlled feature-level failure models for held-out sensing modalities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def corrupt_modality(
    values: pd.DataFrame,
    *,
    kind: str,
    median: pd.Series,
    iqr: pd.Series,
    rng: np.random.Generator,
    severity: float = 1.0,
) -> pd.DataFrame:
    """Return a corrupted copy while preserving naturally missing cells."""
    corrupted = values.copy()
    observed = corrupted.notna()
    scale = iqr.replace(0, 1.0).fillna(1.0)

    if kind == "dropout":
        corrupted.loc[:, :] = np.nan
    elif kind == "partial_dropout":
        remove = pd.DataFrame(
            rng.random(corrupted.shape) < severity,
            index=corrupted.index,
            columns=corrupted.columns,
        )
        corrupted = corrupted.mask(remove & observed)
    elif kind == "gaussian":
        noise = rng.normal(size=corrupted.shape) * scale.to_numpy()[None, :] * severity
        corrupted = corrupted.where(~observed, corrupted + noise)
    elif kind == "bias":
        corrupted = corrupted.where(~observed, corrupted + scale * severity)
    elif kind == "stuck_at_median":
        replacement = pd.DataFrame(
            np.broadcast_to(median.to_numpy(), corrupted.shape),
            index=corrupted.index,
            columns=corrupted.columns,
        )
        corrupted = corrupted.where(~observed, replacement)
    elif kind == "replay":
        order = rng.permutation(len(corrupted))
        replayed = corrupted.iloc[order].copy()
        replayed.index = corrupted.index
        # Severity is the fraction of time windows replaced by stale windows.
        replace = rng.random(len(corrupted)) < severity
        corrupted.loc[replace, :] = replayed.loc[replace, :]
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    return corrupted
