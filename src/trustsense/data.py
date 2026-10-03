"""Loading utilities for the per-participant ExtraSensory release."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import modality_columns


UUID_PATTERN = re.compile(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}", re.IGNORECASE)


@dataclass(frozen=True)
class UserSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def _read_uuid_file(path: Path) -> set[str]:
    uuids = set(UUID_PATTERN.findall(path.read_text(encoding="utf-8")))
    if not uuids:
        raise ValueError(f"No UUIDs found in {path}")
    return {uuid.upper() for uuid in uuids}


def official_test_users(folds_dir: Path, fold: int) -> set[str]:
    """Read Android and iPhone test participants for an official fold."""
    paths = sorted(folds_dir.glob(f"fold_{fold}_test_*_uuids.txt"))
    if len(paths) != 2:
        raise FileNotFoundError(
            f"Expected Android and iPhone test files for fold {fold} under {folds_dir}"
        )
    users: set[str] = set()
    for path in paths:
        users.update(_read_uuid_file(path))
    return users


def nested_official_split(folds_dir: Path, outer_fold: int, validation_fold: int) -> UserSplit:
    """Create train/validation/test users from two disjoint official folds."""
    if outer_fold == validation_fold:
        raise ValueError("Outer test fold and validation fold must differ")

    all_users = set().union(*(official_test_users(folds_dir, fold) for fold in range(5)))
    test = official_test_users(folds_dir, outer_fold)
    validation = official_test_users(folds_dir, validation_fold)
    train = all_users - test - validation

    if train & validation or train & test or validation & test:
        raise AssertionError("User split is not disjoint")
    if len(all_users) != 60:
        raise ValueError(f"Expected 60 unique users, found {len(all_users)}")
    return UserSplit(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
    )


def user_id_from_path(path: Path) -> str:
    match = UUID_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse participant UUID from {path}")
    return match.group(0).upper()


def index_user_files(data_dir: Path) -> dict[str, Path]:
    mapping = {user_id_from_path(path): path for path in data_dir.glob("*.csv.gz")}
    if len(mapping) != 60:
        raise ValueError(f"Expected 60 participant files under {data_dir}, found {len(mapping)}")
    return mapping


def columns_for_experiment(
    data_dir: Path, modalities: list[str], labels: list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    first = next(iter(sorted(data_dir.glob("*.csv.gz"))), None)
    if first is None:
        raise FileNotFoundError(f"No participant files found under {data_dir}")
    columns = pd.read_csv(first, nrows=0).columns.tolist()
    all_modalities = modality_columns(columns)
    unknown_modalities = sorted(set(modalities) - set(all_modalities))
    unknown_labels = sorted(set(labels) - set(columns))
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {unknown_modalities}")
    if unknown_labels:
        raise ValueError(f"Unknown labels: {unknown_labels}")
    selected = {modality: all_modalities[modality] for modality in modalities}
    usecols = ["timestamp", *labels]
    for modality in modalities:
        usecols.extend(selected[modality])
    return list(dict.fromkeys(usecols)), selected


def load_users(
    file_index: dict[str, Path], users: tuple[str, ...], usecols: list[str]
) -> pd.DataFrame:
    """Load a disjoint set of participants and attach participant identity."""
    frames: list[pd.DataFrame] = []
    for user in users:
        frame = pd.read_csv(file_index[user], usecols=usecols, low_memory=False)
        frame.insert(0, "user_id", user)
        numeric = frame.columns.difference(["user_id"])
        frame[numeric] = frame[numeric].astype(np.float32)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def availability(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Whether at least one feature from a modality is present for each row."""
    return frame[columns].notna().any(axis=1).to_numpy(dtype=bool)

