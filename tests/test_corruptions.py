import numpy as np
import pandas as pd

from trustsense.corruptions import corrupt_modality


def test_replay_severity_zero_and_one() -> None:
    values = pd.DataFrame({"a": np.arange(100, dtype=float)})
    kwargs = {
        "kind": "replay",
        "median": values.median(),
        "iqr": values.quantile(0.75) - values.quantile(0.25),
    }
    unchanged = corrupt_modality(values, rng=np.random.default_rng(1), severity=0.0, **kwargs)
    replayed = corrupt_modality(values, rng=np.random.default_rng(1), severity=1.0, **kwargs)
    assert unchanged.equals(values)
    assert not replayed.equals(values)
