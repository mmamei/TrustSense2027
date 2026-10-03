"""Aggregate the four locked folds and compute participant-paired inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METHODS = ("uniform_late_fusion", "reliability_fusion")
METRICS = ("f1", "balanced_accuracy", "brier", "ece", "aurc", "risk_at_80_coverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/full_experiment"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/locked_summary"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_folds(directory: Path, suffix: str) -> pd.DataFrame:
    paths = [directory / f"fold_{fold}_{suffix}.csv" for fold in range(1, 5)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing locked-fold files: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame[frame["method"].isin(METHODS)].copy()


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    sorted_p = p_values.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_p * (len(sorted_p) - np.arange(len(sorted_p))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def stratified_bootstrap_ci(
    values: pd.DataFrame, *, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    fold_arrays = [
        group["delta"].dropna().to_numpy(dtype=float)
        for _, group in values.groupby("fold", sort=True)
    ]
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        fold_means = [
            rng.choice(array, size=len(array), replace=True).mean()
            for array in fold_arrays
            if len(array)
        ]
        draws[index] = np.mean(fold_means)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = load_folds(args.input_dir, "metrics")
    user_metrics = load_folds(args.input_dir, "user_metrics")

    per_fold = (
        fold_metrics.groupby(["fold", "condition", "method"], as_index=False)[list(METRICS)]
        .mean()
    )
    condition_summary = (
        per_fold.groupby(["condition", "method"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    condition_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in condition_summary.columns
    ]
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)

    target_summary = (
        fold_metrics.groupby(["condition", "target_modality", "method"], as_index=False)[
            list(METRICS)
        ]
        .mean()
    )
    target_summary.to_csv(args.output_dir / "target_summary.csv", index=False)

    per_user = (
        user_metrics.groupby(["fold", "user_id", "condition", "method"], as_index=False)[
            list(METRICS)
        ]
        .mean()
    )
    rng = np.random.default_rng(args.seed)
    effects: list[dict[str, float | str | int]] = []
    for condition in sorted(per_user["condition"].unique()):
        subset = per_user[per_user["condition"].eq(condition)]
        for metric in ("f1", "balanced_accuracy", "brier", "ece"):
            pivot = subset.pivot(
                index=["fold", "user_id"], columns="method", values=metric
            ).dropna()
            delta = pivot["reliability_fusion"] - pivot["uniform_late_fusion"]
            paired = delta.rename("delta").reset_index()
            low, high = stratified_bootstrap_ci(
                paired, iterations=args.bootstrap, rng=rng
            )
            try:
                test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                p_value = float(test.pvalue)
            except ValueError:
                p_value = 1.0
            effects.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "participants": len(delta),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilcoxon_p": p_value,
                }
            )

    paired_effects = pd.DataFrame(effects)
    paired_effects["holm_p"] = np.nan
    for metric, indices in paired_effects.groupby("metric").groups.items():
        paired_effects.loc[indices, "holm_p"] = holm_adjust(
            paired_effects.loc[indices, "wilcoxon_p"]
        )
    paired_effects.to_csv(args.output_dir / "paired_user_effects.csv", index=False)

    f1_effects = paired_effects[paired_effects["metric"].eq("f1")].copy()
    lines = [
        "# Locked evaluation summary",
        "",
        "Fold 0 was used for method development. This summary contains only official folds 1–4,",
        "covering 48 distinct test participants.",
        "",
        "## Participant-paired macro-F1 effects",
        "",
        "| Condition | Mean delta | 95% stratified bootstrap CI | Holm-adjusted p |",
        "|---|---:|---:|---:|",
    ]
    for row in f1_effects.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.mean_delta:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | {row.holm_p:.4g} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

