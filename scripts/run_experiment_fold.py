"""Run one nested official fold with calibration and validation-only tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from trustsense.calibration import PlattCalibrator
from trustsense.corruptions import corrupt_modality
from trustsense.data import columns_for_experiment, index_user_files, load_users, nested_official_split
from trustsense.fusion import (
    base_reliability,
    fit_reliability_profile,
    reliability_fusion,
    uniform_fusion,
)
from trustsense.gating import (
    fit_integrity_gate,
    gated_fusion,
    integrity_probability,
    probability_tensor,
)
from trustsense.metrics import binary_metrics, optimal_f1_threshold
from trustsense.models import make_logistic_sgd


CORRUPTIONS = {
    "partial_dropout": 0.5,
    "gaussian": 2.0,
    "bias": 2.0,
    "replay": 1.0,
}
TUNING_MODALITIES = {"phone_acc", "watch_acc", "audio"}
FLOORS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRENGTHS = (0.0, 2.0, 4.0, 8.0)


def selective_risk(
    truth: np.ndarray, probability: np.ndarray, confidence: np.ndarray
) -> float:
    order = np.argsort(-confidence, kind="stable")
    errors = ((probability >= 0.5).astype(int) != truth.astype(int)).astype(float)
    cumulative = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(cumulative.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/extrasensory"))
    parser.add_argument(
        "--folds-dir", type=Path, default=Path("data/raw/cv5Folds/cv_5_folds")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_experiment"))
    parser.add_argument(
        "--locked",
        action="store_true",
        help="Use the fold-0-frozen heuristic and skip learned-gate development.",
    )
    return parser.parse_args()


def calibrated_predict(model: Any, calibrator: PlattCalibrator, values: pd.DataFrame) -> np.ndarray:
    return calibrator.predict(model.decision_function(values))


def scenario_state(
    *,
    frame: pd.DataFrame,
    kind: str,
    target: str | None,
    severity: float,
    modalities: list[str],
    labels: list[str],
    columns_by_modality: dict[str, list[str]],
    profiles: dict[str, Any],
    models: dict[tuple[str, str], Any],
    calibrators: dict[tuple[str, str], PlattCalibrator],
    clean_probability: dict[tuple[str, str], np.ndarray],
    clean_base: np.ndarray,
    seed: int,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    probability = dict(clean_probability)
    base = clean_base.copy()
    if target is None:
        return probability, base

    columns = columns_by_modality[target]
    profile = profiles[target]
    corrupted = corrupt_modality(
        frame[columns],
        kind=kind,
        median=profile.median,
        iqr=profile.iqr,
        rng=np.random.default_rng(seed),
        severity=severity,
    )
    modality_index = modalities.index(target)
    base[:, modality_index] = base_reliability(corrupted, profile)
    for label in labels:
        key = (target, label)
        probability[key] = calibrated_predict(models[key], calibrators[key], corrupted)
    return probability, base


def stacked_for_label(
    probability: dict[tuple[str, str], np.ndarray], modalities: list[str], label: str
) -> np.ndarray:
    return np.column_stack([probability[(modality, label)] for modality in modalities])


def tune_reliability(
    *,
    validation: pd.DataFrame,
    scenarios: list[tuple[str, str | None, dict[tuple[str, str], np.ndarray], np.ndarray]],
    modalities: list[str],
    labels: list[str],
) -> tuple[float, float, list[dict[str, float]]]:
    rng = np.random.default_rng(2027)
    sampled_indices: list[int] = []
    for _, group in validation.groupby("user_id", sort=True):
        indices = group.index.to_numpy()
        take = min(1250, len(indices))
        sampled_indices.extend(rng.choice(indices, size=take, replace=False).tolist())
    sampled = np.array(sorted(sampled_indices), dtype=int)
    sampled_truth = {
        label: validation.iloc[sampled][label].to_numpy() for label in labels
    }

    clean = next(item for item in scenarios if item[0] == "clean")
    clean_uniform_f1: list[float] = []
    for label in labels:
        probability = stacked_for_label(clean[2], modalities, label)[sampled]
        fused = uniform_fusion(probability, clean[3][sampled])
        truth_all = sampled_truth[label]
        observed = ~np.isnan(truth_all)
        clean_uniform_f1.append(
            f1_score(
                truth_all[observed].astype(int),
                fused[observed] >= 0.5,
                zero_division=0,
            )
        )
    clean_floor = float(np.mean(clean_uniform_f1)) - 0.005

    trials: list[dict[str, float]] = []
    for floor in FLOORS:
        for strength in STRENGTHS:
            selective_losses: list[float] = []
            candidate_clean_f1: list[float] = []
            for kind, _, probability_by_key, base in scenarios:
                for label in labels:
                    probability = stacked_for_label(
                        probability_by_key, modalities, label
                    )[sampled]
                    fused, evidence = reliability_fusion(
                        probability,
                        base[sampled],
                        disagreement_strength=strength,
                        reliability_floor=floor,
                    )
                    truth_all = sampled_truth[label]
                    observed = ~np.isnan(truth_all)
                    truth = truth_all[observed].astype(float)
                    available_count = np.maximum((base[sampled] > 0).sum(axis=1), 1)
                    confidence = (
                        2.0
                        * np.abs(fused - 0.5)
                        * np.clip(evidence / available_count, 0.0, 1.0)
                    )
                    selective_losses.append(
                        selective_risk(truth, fused[observed], confidence[observed])
                    )
                    if kind == "clean":
                        candidate_clean_f1.append(
                            f1_score(truth.astype(int), fused[observed] >= 0.5, zero_division=0)
                        )
            clean_f1 = float(np.mean(candidate_clean_f1))
            trials.append(
                {
                    "floor": floor,
                    "strength": strength,
                    "validation_aurc": float(np.mean(selective_losses)),
                    "clean_f1": clean_f1,
                    "eligible": float(clean_f1 >= clean_floor),
                }
            )

    eligible = [trial for trial in trials if trial["eligible"] == 1.0]
    chosen = min(eligible or trials, key=lambda trial: trial["validation_aurc"])
    return chosen["floor"], chosen["strength"], trials


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    modalities: list[str] = config["modalities"]
    labels: list[str] = config["labels"]
    validation_fold = (args.outer_fold + 1) % 5
    split = nested_official_split(args.folds_dir, args.outer_fold, validation_fold)
    usecols, columns_by_modality = columns_for_experiment(args.data_dir, modalities, labels)
    files = index_user_files(args.data_dir)
    train = load_users(files, split.train, usecols)
    validation = load_users(files, split.validation, usecols)
    test = load_users(files, split.test, usecols)
    print(
        f"Fold {args.outer_fold}; validation fold {validation_fold}; "
        f"rows {len(train)}/{len(validation)}/{len(test)}",
        flush=True,
    )

    profiles = {
        modality: fit_reliability_profile(train[columns])
        for modality, columns in columns_by_modality.items()
    }
    validation_base = np.column_stack(
        [
            base_reliability(validation[columns_by_modality[modality]], profiles[modality])
            for modality in modalities
        ]
    )
    test_base = np.column_stack(
        [
            base_reliability(test[columns_by_modality[modality]], profiles[modality])
            for modality in modalities
        ]
    )

    models: dict[tuple[str, str], Any] = {}
    calibrators: dict[tuple[str, str], PlattCalibrator] = {}
    validation_clean: dict[tuple[str, str], np.ndarray] = {}
    test_clean: dict[tuple[str, str], np.ndarray] = {}
    for modality in modalities:
        columns = columns_by_modality[modality]
        print(f"Training {modality}", flush=True)
        for label in labels:
            train_observed = train[label].notna().to_numpy()
            validation_observed = validation[label].notna().to_numpy()
            key = (modality, label)
            model = make_logistic_sgd(seed=config["seed"] + args.outer_fold, **config["model"])
            model.fit(
                train.loc[train_observed, columns],
                train.loc[train_observed, label].astype(int),
            )
            calibrator = PlattCalibrator().fit(
                model.decision_function(validation.loc[validation_observed, columns]),
                validation.loc[validation_observed, label].to_numpy(dtype=int),
            )
            models[key] = model
            calibrators[key] = calibrator
            validation_clean[key] = calibrated_predict(model, calibrator, validation[columns])
            test_clean[key] = calibrated_predict(model, calibrator, test[columns])

    conditions: list[tuple[str, str | None, float]] = [("clean", None, 0.0)]
    conditions.extend(
        (kind, modality, severity)
        for kind, severity in CORRUPTIONS.items()
        for modality in modalities
    )

    tuning_conditions = (
        [conditions[0]]
        if args.locked
        else [
            condition
            for condition in conditions
            if condition[1] is None or condition[1] in TUNING_MODALITIES
        ]
    )
    validation_scenarios = []
    for index, (kind, target, severity) in enumerate(tuning_conditions):
        print(
            f"Validation scenario {index + 1}/{len(tuning_conditions)}: {kind}/{target or 'none'}",
            flush=True,
        )
        probability, base = scenario_state(
            frame=validation,
            kind=kind,
            target=target,
            severity=severity,
            modalities=modalities,
            labels=labels,
            columns_by_modality=columns_by_modality,
            profiles=profiles,
            models=models,
            calibrators=calibrators,
            clean_probability=validation_clean,
            clean_base=validation_base,
            seed=config["seed"] + args.outer_fold * 1000 + index,
        )
        validation_scenarios.append((kind, target, probability, base))

    clean_validation_state = validation_scenarios[0]
    floor, strength = 0.25, 4.0
    gate = None
    sampled = np.array([], dtype=int)
    corrupted_gate_states: list[tuple[int, np.ndarray, np.ndarray]] = []
    if not args.locked:
        rng = np.random.default_rng(config["seed"] + args.outer_fold)
        sampled_indices: list[int] = []
        for _, group in validation.groupby("user_id", sort=True):
            indices = group.index.to_numpy()
            sampled_indices.extend(
                rng.choice(indices, size=min(1250, len(indices)), replace=False).tolist()
            )
        sampled = np.array(sorted(sampled_indices), dtype=int)
        clean_gate_probability = probability_tensor(
            clean_validation_state[2], modalities, labels
        )[sampled]
        for _, target, probability_by_key, base in validation_scenarios[1:]:
            if target is None:
                continue
            corrupted_gate_states.append(
                (
                    modalities.index(target),
                    probability_tensor(probability_by_key, modalities, labels)[sampled],
                    base[sampled],
                )
            )
        gate = fit_integrity_gate(
            clean_gate_probability,
            clean_validation_state[3][sampled],
            corrupted_gate_states,
            seed=config["seed"] + args.outer_fold,
        )
        print("Fitted learned integrity gate", flush=True)

    validation_uniform: dict[str, np.ndarray] = {}
    validation_reliable: dict[str, np.ndarray] = {}
    validation_learned: dict[str, np.ndarray] = {}
    clean_validation_gated = None
    if gate is not None:
        clean_validation_tensor = probability_tensor(
            clean_validation_state[2], modalities, labels
        )
        clean_validation_integrity = integrity_probability(
            gate, clean_validation_tensor, clean_validation_state[3]
        )
        clean_validation_gated, _ = gated_fusion(
            clean_validation_tensor, clean_validation_integrity
        )
    for label in labels:
        label_index = labels.index(label)
        stacked = stacked_for_label(clean_validation_state[2], modalities, label)
        validation_uniform[label] = uniform_fusion(stacked, clean_validation_state[3])
        validation_reliable[label], _ = reliability_fusion(
            stacked,
            clean_validation_state[3],
            disagreement_strength=strength,
            reliability_floor=floor,
        )
        if clean_validation_gated is not None:
            validation_learned[label] = clean_validation_gated[:, label_index]
    thresholds = {
        "uniform_late_fusion": {
            label: optimal_f1_threshold(
                validation[label].to_numpy(), validation_uniform[label]
            )
            for label in labels
        },
        "reliability_fusion": {
            label: optimal_f1_threshold(
                validation[label].to_numpy(), validation_reliable[label]
            )
            for label in labels
        },
    }
    if validation_learned:
        thresholds["learned_reliability_fusion"] = {
            label: optimal_f1_threshold(
                validation[label].to_numpy(), validation_learned[label]
            )
            for label in labels
        }

    records: list[dict[str, object]] = []
    user_records: list[dict[str, object]] = []
    for index, (kind, target, severity) in enumerate(conditions):
        if index % 4 == 0:
            print(
                f"Test scenario {index + 1}/{len(conditions)}: {kind}/{target or 'none'}",
                flush=True,
            )
        probability, base = scenario_state(
            frame=test,
            kind=kind,
            target=target,
            severity=severity,
            modalities=modalities,
            labels=labels,
            columns_by_modality=columns_by_modality,
            profiles=profiles,
            models=models,
            calibrators=calibrators,
            clean_probability=test_clean,
            clean_base=test_base,
            seed=config["seed"] + args.outer_fold * 1000 + index,
        )
        available_count = np.maximum((base > 0).sum(axis=1), 1)
        learned_all = None
        integrity = None
        if gate is not None:
            tensor = probability_tensor(probability, modalities, labels)
            integrity = integrity_probability(gate, tensor, base)
            learned_all, _ = gated_fusion(tensor, integrity)
        for label in labels:
            label_index = labels.index(label)
            stacked = stacked_for_label(probability, modalities, label)
            uniform = uniform_fusion(stacked, base)
            reliable, evidence = reliability_fusion(
                stacked,
                base,
                disagreement_strength=strength,
                reliability_floor=floor,
            )
            outputs = [
                (
                    "uniform_late_fusion",
                    uniform,
                    2.0 * np.abs(uniform - 0.5),
                ),
                (
                    "reliability_fusion",
                    reliable,
                    2.0
                    * np.abs(reliable - 0.5)
                    * np.clip(
                        evidence / np.maximum((base > 0).sum(axis=1), 1),
                        0.0,
                        1.0,
                    ),
                ),
            ]
            if learned_all is not None and integrity is not None:
                learned = learned_all[:, label_index]
                outputs.append(
                    (
                        "learned_reliability_fusion",
                        learned,
                        2.0
                        * np.abs(learned - 0.5)
                        * np.clip(
                            integrity.sum(axis=1) / available_count,
                            0.0,
                            1.0,
                        ),
                    )
                )
            for method, predicted, confidence in outputs:
                common = {
                    "fold": args.outer_fold,
                    "validation_fold": validation_fold,
                    "condition": kind,
                    "target_modality": target or "none",
                    "severity": severity,
                    "method": method,
                    "label": label,
                    "threshold": thresholds[method][label],
                    "reliability_floor": floor,
                    "disagreement_strength": strength,
                }
                records.append(
                    {
                        **common,
                        **binary_metrics(
                            test[label].to_numpy(),
                            predicted,
                            confidence=confidence,
                            threshold=thresholds[method][label],
                        ),
                    }
                )
                for user_id in split.test:
                    user_mask = test["user_id"].eq(user_id).to_numpy()
                    user_records.append(
                        {
                            **common,
                            "user_id": user_id,
                            **binary_metrics(
                                test.loc[user_mask, label].to_numpy(),
                                predicted[user_mask],
                                confidence=confidence[user_mask],
                                threshold=thresholds[method][label],
                            ),
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / f"fold_{args.outer_fold}_metrics.csv", index=False)
    pd.DataFrame(user_records).to_csv(
        args.output_dir / f"fold_{args.outer_fold}_user_metrics.csv", index=False
    )
    metadata = {
        "outer_fold": args.outer_fold,
        "validation_fold": validation_fold,
        "train_users": split.train,
        "validation_users": split.validation,
        "test_users": split.test,
        "reliability_floor": floor,
        "disagreement_strength": strength,
        "gate_training_rows": int(len(sampled)),
        "gate_training_scenarios": len(corrupted_gate_states),
        "locked_evaluation": args.locked,
        "thresholds": thresholds,
    }
    (args.output_dir / f"fold_{args.outer_fold}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Completed fold {args.outer_fold}", flush=True)


if __name__ == "__main__":
    main()
