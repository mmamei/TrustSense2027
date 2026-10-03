import numpy as np
import pandas as pd

from trustsense.fusion import (
    base_reliability,
    fit_reliability_profile,
    reliability_components,
    reliability_fusion,
    uniform_fusion,
)


def test_uniform_fusion_ignores_unavailable_modality() -> None:
    probabilities = np.array([[0.2, 0.8, 0.5]])
    reliability = np.array([[1.0, 0.0, 1.0]])
    assert np.allclose(uniform_fusion(probabilities, reliability), [0.35])


def test_reliability_fusion_downweights_disagreement() -> None:
    probabilities = np.array([[0.1, 0.2, 0.95]])
    base = np.ones_like(probabilities)
    uniform = uniform_fusion(probabilities, base)
    reliable, evidence = reliability_fusion(
        probabilities, base, disagreement_strength=8.0, reliability_floor=0.25
    )
    assert reliable[0] < uniform[0]
    assert evidence[0] > 0


def test_reliability_components_reconstruct_base_score() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]})
    values = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})
    profile = fit_reliability_profile(train)
    completeness, ood = reliability_components(values, profile)
    assert np.allclose(base_reliability(values, profile), completeness * ood)
    assert completeness[1] == 0.0
