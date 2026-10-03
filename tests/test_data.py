from pathlib import Path

from trustsense.data import nested_official_split


def test_official_nested_split_is_disjoint_and_complete() -> None:
    folds = Path("data/raw/cv5Folds/cv_5_folds")
    if not folds.exists():
        return
    split = nested_official_split(folds, outer_fold=0, validation_fold=1)
    assert len(split.train) == 36
    assert len(split.validation) == 12
    assert len(split.test) == 12
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))

