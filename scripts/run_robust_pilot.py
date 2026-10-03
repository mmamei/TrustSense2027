"""Evaluate uniform and reliability-aware fusion under controlled faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustsense.corruptions import corrupt_modality
from trustsense.data import (
    columns_for_experiment,
    index_user_files,
    load_users,
    nested_official_split,
)
from trustsense.fusion import (
    base_reliability,
    fit_reliability_profile,
    reliability_fusion,
    uniform_fusion,
)
from trustsense.metrics import binary_metrics
from trustsense.models import make_logistic_sgd


CORRUPTIONS = {
    "dropout": 1.0,
    "partial_dropout": 0.5,
    "gaussian": 2.0,
    "bias": 2.0,
    "stuck_at_median": 1.0,
    "replay": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extrasensory"))
    parser.add_argument(
        "--folds-dir", type=Path, default=Path("data/raw/cv5Folds/cv_5_folds")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/robust_pilot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    modalities: list[str] = config["modalities"]
    labels: list[str] = config["labels"]
    split = nested_official_split(
        args.folds_dir, config["outer_fold"], config["validation_fold"]
    )
    usecols, columns_by_modality = columns_for_experiment(args.data_dir, modalities, labels)
    files = index_user_files(args.data_dir)
    train = load_users(files, split.train, usecols)
    validation = load_users(files, split.validation, usecols)
    test = load_users(files, split.test, usecols)

    profiles = {
        modality: fit_reliability_profile(train[columns])
        for modality, columns in columns_by_modality.items()
    }
    clean_base = np.column_stack(
        [
            base_reliability(test[columns_by_modality[modality]], profiles[modality])
            for modality in modalities
        ]
    )

    models: dict[tuple[str, str], object] = {}
    clean_probability: dict[tuple[str, str], np.ndarray] = {}
    for modality in modalities:
        columns = columns_by_modality[modality]
        print(f"Training {modality}", flush=True)
        for label in labels:
            observed = train[label].notna().to_numpy()
            model = make_logistic_sgd(seed=config["seed"], **config["model"])
            model.fit(train.loc[observed, columns], train.loc[observed, label].astype(int))
            models[(modality, label)] = model
            clean_probability[(modality, label)] = model.predict_proba(test[columns])[:, 1]

    conditions: list[tuple[str, str | None, float]] = [("clean", None, 0.0)]
    conditions.extend(
        (kind, modality, severity)
        for kind, severity in CORRUPTIONS.items()
        for modality in modalities
    )

    rng = np.random.default_rng(config["seed"])
    records: list[dict[str, object]] = []
    for kind, target, severity in conditions:
        probability = dict(clean_probability)
        base = clean_base.copy()
        if target is not None:
            columns = columns_by_modality[target]
            profile = profiles[target]
            corrupted = corrupt_modality(
                test[columns],
                kind=kind,
                median=profile.median,
                iqr=profile.iqr,
                rng=rng,
                severity=severity,
            )
            modality_index = modalities.index(target)
            base[:, modality_index] = base_reliability(corrupted, profile)
            for label in labels:
                probability[(target, label)] = models[(target, label)].predict_proba(corrupted)[:, 1]

        for label in labels:
            stacked = np.column_stack(
                [probability[(modality, label)] for modality in modalities]
            )
            uniform = uniform_fusion(stacked, base)
            reliable, evidence = reliability_fusion(stacked, base)
            confidence_uniform = 2.0 * np.abs(uniform - 0.5)
            confidence_reliable = (
                2.0
                * np.abs(reliable - 0.5)
                * np.clip(evidence / len(modalities), 0.0, 1.0)
            )
            for method, predicted, confidence in (
                ("uniform_late_fusion", uniform, confidence_uniform),
                ("reliability_fusion", reliable, confidence_reliable),
            ):
                records.append(
                    {
                        "condition": kind,
                        "target_modality": target or "none",
                        "severity": severity,
                        "method": method,
                        "label": label,
                        "mean_evidence": float(np.mean(evidence)) if method == "reliability_fusion" else np.nan,
                        **binary_metrics(
                            test[label].to_numpy(), predicted, confidence=confidence
                        ),
                    }
                )

    results = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"fold_{config['outer_fold']}_metrics.csv"
    results.to_csv(output, index=False)

    summary = (
        results.groupby(["condition", "method"], as_index=False)[
            [
                "f1",
                "balanced_accuracy",
                "brier",
                "ece",
                "aurc",
                "risk_at_80_coverage",
            ]
        ]
        .mean()
        .sort_values(["condition", "method"])
    )
    print(f"Wrote {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
