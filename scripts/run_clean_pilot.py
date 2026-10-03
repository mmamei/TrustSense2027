"""Run one clean official fold to validate the late-fusion experiment design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustsense.data import (
    availability,
    columns_for_experiment,
    index_user_files,
    load_users,
    nested_official_split,
)
from trustsense.metrics import binary_metrics
from trustsense.models import make_logistic_sgd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extrasensory"))
    parser.add_argument(
        "--folds-dir", type=Path, default=Path("data/raw/cv5Folds/cv_5_folds")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/clean_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    modalities: list[str] = config["modalities"]
    labels: list[str] = config["labels"]
    split = nested_official_split(
        args.folds_dir, config["outer_fold"], config["validation_fold"]
    )
    usecols, modality_columns = columns_for_experiment(args.data_dir, modalities, labels)
    files = index_user_files(args.data_dir)

    # Validation is loaded now because it will be used for calibration and trust
    # hyperparameters in the full experiment. The clean pilot intentionally does
    # not tune on it.
    train = load_users(files, split.train, usecols)
    validation = load_users(files, split.validation, usecols)
    test = load_users(files, split.test, usecols)

    print(
        f"Users train/validation/test: {len(split.train)}/{len(split.validation)}/{len(split.test)}"
    )
    print(f"Rows train/validation/test: {len(train)}/{len(validation)}/{len(test)}")

    probability: dict[tuple[str, str], np.ndarray] = {}
    availability_by_modality = {
        modality: availability(test, columns) for modality, columns in modality_columns.items()
    }
    records: list[dict[str, object]] = []

    for modality, columns in modality_columns.items():
        print(f"Training modality: {modality} ({len(columns)} features)")
        x_train = train[columns]
        x_test = test[columns]
        for label in labels:
            observed_train = train[label].notna().to_numpy()
            y_train = train.loc[observed_train, label].to_numpy(dtype=int)
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"Training split has fewer than two classes for {label}")
            estimator = make_logistic_sgd(seed=config["seed"], **config["model"])
            estimator.fit(x_train.loc[observed_train], y_train)
            prob = estimator.predict_proba(x_test)[:, 1]
            probability[(modality, label)] = prob
            records.append(
                {
                    "method": modality,
                    "label": label,
                    **binary_metrics(test[label].to_numpy(), prob),
                }
            )

    for label in labels:
        probs = np.column_stack([probability[(modality, label)] for modality in modalities])
        present = np.column_stack([availability_by_modality[modality] for modality in modalities])
        weights = present.astype(float)
        denominator = weights.sum(axis=1)
        fallback = probs.mean(axis=1)
        fused = np.divide(
            (probs * weights).sum(axis=1),
            denominator,
            out=fallback.copy(),
            where=denominator > 0,
        )
        records.append(
            {
                "method": "uniform_late_fusion",
                "label": label,
                **binary_metrics(test[label].to_numpy(), fused),
            }
        )

    results = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / f"fold_{config['outer_fold']}_metrics.csv", index=False)
    split_payload = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    (args.output_dir / f"fold_{config['outer_fold']}_users.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    summary = (
        results.groupby("method", as_index=False)[
            ["f1", "balanced_accuracy", "brier", "ece", "aurc", "risk_at_80_coverage"]
        ]
        .mean()
        .sort_values("f1", ascending=False)
    )
    print("\nMacro-average across selected labels:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
