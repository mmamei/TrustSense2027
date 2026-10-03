"""Locked-fold ablations, severity sweeps, and clean unimodal baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustsense.calibration import PlattCalibrator
from trustsense.corruptions import corrupt_modality
from trustsense.data import columns_for_experiment, index_user_files, load_users, nested_official_split
from trustsense.fusion import (
    fit_reliability_profile,
    reliability_components,
    reliability_fusion,
    uniform_fusion,
)
from trustsense.metrics import binary_metrics, optimal_f1_threshold
from trustsense.models import make_logistic_sgd


STANDARD = {"partial_dropout": 0.5, "gaussian": 2.0, "bias": 2.0, "replay": 1.0}
SWEEPS = {
    "partial_dropout": (0.10, 0.25, 0.50, 0.75),
    "gaussian": (0.50, 1.00, 2.00, 4.00),
    "bias": (0.50, 1.00, 2.00, 4.00),
    "replay": (0.25, 0.50, 0.75, 1.00),
}
SWEEP_MODALITIES = ("phone_acc", "watch_acc", "audio")
FLOOR = 0.25
STRENGTH = 4.0


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(1, 5))
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extrasensory"))
    parser.add_argument("--folds-dir", type=Path, default=Path("data/raw/cv5Folds/cv_5_folds"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/extended_evaluation"))
    return parser.parse_args()


def predict(model: Any, calibrator: PlattCalibrator, values: pd.DataFrame) -> np.ndarray:
    return calibrator.predict(model.decision_function(values))


def stack(probability: dict[tuple[str, str], np.ndarray], modalities: list[str], label: str) -> np.ndarray:
    return np.column_stack([probability[(modality, label)] for modality in modalities])


def weighted(probability: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    fallback = probability.mean(axis=1)
    return np.divide(
        (probability * weights).sum(axis=1), denominator,
        out=fallback.copy(), where=denominator > 0,
    )


def method_outputs(probability: np.ndarray, completeness: np.ndarray, ood: np.ndarray) -> dict[str, np.ndarray]:
    availability = (completeness > 0).astype(float)
    base = completeness * ood
    full, _ = reliability_fusion(
        probability, base, reliability_floor=FLOOR, disagreement_strength=STRENGTH
    )
    agreement, _ = reliability_fusion(
        probability, availability, reliability_floor=FLOOR, disagreement_strength=STRENGTH
    )
    return {
        "uniform": uniform_fusion(probability, base),
        "completeness_only": weighted(probability, completeness),
        "ood_only": weighted(probability, availability * ood),
        "agreement_only": agreement,
        "full_reliability": full,
    }


def main() -> None:
    options = args()
    config = json.loads(options.config.read_text(encoding="utf-8"))
    modalities: list[str] = config["modalities"]
    labels: list[str] = config["labels"]
    split = nested_official_split(options.folds_dir, options.outer_fold, (options.outer_fold + 1) % 5)
    usecols, columns_by_modality = columns_for_experiment(options.data_dir, modalities, labels)
    files = index_user_files(options.data_dir)
    train = load_users(files, split.train, usecols)
    validation = load_users(files, split.validation, usecols)
    test = load_users(files, split.test, usecols)
    print(f"Fold {options.outer_fold}: rows {len(train)}/{len(validation)}/{len(test)}", flush=True)

    profiles = {m: fit_reliability_profile(train[columns_by_modality[m]]) for m in modalities}
    clean_components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    validation_components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for modality in modalities:
        validation_components[modality] = reliability_components(
            validation[columns_by_modality[modality]], profiles[modality]
        )
        clean_components[modality] = reliability_components(
            test[columns_by_modality[modality]], profiles[modality]
        )
    validation_completeness = np.column_stack([validation_components[m][0] for m in modalities])
    validation_ood = np.column_stack([validation_components[m][1] for m in modalities])
    clean_completeness = np.column_stack([clean_components[m][0] for m in modalities])
    clean_ood = np.column_stack([clean_components[m][1] for m in modalities])

    models: dict[tuple[str, str], Any] = {}
    calibrators: dict[tuple[str, str], PlattCalibrator] = {}
    validation_probability: dict[tuple[str, str], np.ndarray] = {}
    clean_probability: dict[tuple[str, str], np.ndarray] = {}
    for modality in modalities:
        print(f"Training {modality}", flush=True)
        columns = columns_by_modality[modality]
        for label in labels:
            train_observed = train[label].notna().to_numpy()
            validation_observed = validation[label].notna().to_numpy()
            model = make_logistic_sgd(seed=config["seed"] + options.outer_fold, **config["model"])
            model.fit(train.loc[train_observed, columns], train.loc[train_observed, label].astype(int))
            calibrator = PlattCalibrator().fit(
                model.decision_function(validation.loc[validation_observed, columns]),
                validation.loc[validation_observed, label].to_numpy(dtype=int),
            )
            key = (modality, label)
            models[key], calibrators[key] = model, calibrator
            validation_probability[key] = predict(model, calibrator, validation[columns])
            clean_probability[key] = predict(model, calibrator, test[columns])

    methods = tuple(method_outputs(
        stack(validation_probability, modalities, labels[0]), validation_completeness, validation_ood
    ))
    thresholds: dict[str, dict[str, float]] = {method: {} for method in methods}
    for label in labels:
        outputs = method_outputs(
            stack(validation_probability, modalities, label), validation_completeness, validation_ood
        )
        for method, values in outputs.items():
            thresholds[method][label] = optimal_f1_threshold(validation[label].to_numpy(), values)
    unimodal_thresholds = {
        modality: {
            label: optimal_f1_threshold(validation[label].to_numpy(), validation_probability[(modality, label)])
            for label in labels
        }
        for modality in modalities
    }

    ablations: list[dict[str, object]] = []
    severity_rows: list[dict[str, object]] = []
    unimodal: list[dict[str, object]] = []

    def record_outputs(
        destination: list[dict[str, object]], *, kind: str, target: str, severity: float,
        probability_by_key: dict[tuple[str, str], np.ndarray], completeness: np.ndarray,
        ood: np.ndarray, selected_methods: tuple[str, ...],
    ) -> None:
        for label in labels:
            outputs = method_outputs(stack(probability_by_key, modalities, label), completeness, ood)
            for method in selected_methods:
                destination.append({
                    "fold": options.outer_fold, "condition": kind, "target_modality": target,
                    "severity": severity, "method": method, "label": label,
                    **binary_metrics(test[label].to_numpy(), outputs[method], threshold=thresholds[method][label]),
                })

    record_outputs(
        ablations, kind="clean", target="none", severity=0.0,
        probability_by_key=clean_probability, completeness=clean_completeness, ood=clean_ood,
        selected_methods=methods,
    )
    for modality in modalities:
        for label in labels:
            unimodal.append({
                "fold": options.outer_fold, "modality": modality, "label": label,
                **binary_metrics(
                    test[label].to_numpy(), clean_probability[(modality, label)],
                    threshold=unimodal_thresholds[modality][label],
                ),
            })

    scenarios = [(kind, severity, modality) for kind, levels in SWEEPS.items()
                 for severity in levels for modality in SWEEP_MODALITIES]
    for index, (kind, severity, target) in enumerate(scenarios, start=1):
        print(f"Scenario {index}/{len(scenarios)}: {kind}/{severity:g}/{target}", flush=True)
        probability_by_key = dict(clean_probability)
        completeness = clean_completeness.copy()
        ood = clean_ood.copy()
        columns = columns_by_modality[target]
        profile = profiles[target]
        corrupted = corrupt_modality(
            test[columns], kind=kind, median=profile.median, iqr=profile.iqr,
            rng=np.random.default_rng(config["seed"] + options.outer_fold * 10000 + index),
            severity=severity,
        )
        target_index = modalities.index(target)
        completeness[:, target_index], ood[:, target_index] = reliability_components(corrupted, profile)
        for label in labels:
            key = (target, label)
            probability_by_key[key] = predict(models[key], calibrators[key], corrupted)
        record_outputs(
            severity_rows, kind=kind, target=target, severity=severity,
            probability_by_key=probability_by_key, completeness=completeness, ood=ood,
            selected_methods=("uniform", "full_reliability"),
        )
        if np.isclose(severity, STANDARD[kind]):
            record_outputs(
                ablations, kind=kind, target=target, severity=severity,
                probability_by_key=probability_by_key, completeness=completeness, ood=ood,
                selected_methods=methods,
            )

    # Add the remaining three modalities to the standard-severity ablation.
    for index, (kind, severity) in enumerate(STANDARD.items(), start=1):
        for target in [m for m in modalities if m not in SWEEP_MODALITIES]:
            probability_by_key = dict(clean_probability)
            completeness = clean_completeness.copy()
            ood = clean_ood.copy()
            columns = columns_by_modality[target]
            profile = profiles[target]
            corrupted = corrupt_modality(
                test[columns], kind=kind, median=profile.median, iqr=profile.iqr,
                rng=np.random.default_rng(config["seed"] + options.outer_fold * 20000 + index * 10 + modalities.index(target)),
                severity=severity,
            )
            target_index = modalities.index(target)
            completeness[:, target_index], ood[:, target_index] = reliability_components(corrupted, profile)
            for label in labels:
                key = (target, label)
                probability_by_key[key] = predict(models[key], calibrators[key], corrupted)
            record_outputs(
                ablations, kind=kind, target=target, severity=severity,
                probability_by_key=probability_by_key, completeness=completeness, ood=ood,
                selected_methods=methods,
            )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablations).to_csv(options.output_dir / f"fold_{options.outer_fold}_ablation.csv", index=False)
    pd.DataFrame(severity_rows).to_csv(options.output_dir / f"fold_{options.outer_fold}_severity.csv", index=False)
    pd.DataFrame(unimodal).to_csv(options.output_dir / f"fold_{options.outer_fold}_unimodal.csv", index=False)
    (options.output_dir / f"fold_{options.outer_fold}_metadata.json").write_text(
        json.dumps({"fold": options.outer_fold, "locked": True, "sweeps": SWEEPS,
                    "sweep_modalities": SWEEP_MODALITIES, "floor": FLOOR, "strength": STRENGTH,
                    "thresholds": thresholds}, indent=2), encoding="utf-8"
    )
    print(f"Completed extended fold {options.outer_fold}", flush=True)


if __name__ == "__main__":
    main()
