"""Aggregate the locked extended evaluation at fold level."""

from pathlib import Path

import pandas as pd


INPUT = Path("results/extended_evaluation")
OUTPUT = Path("results/extended_summary")
METRICS = ["f1", "balanced_accuracy", "brier", "ece"]


def load(kind: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(INPUT / f"fold_{fold}_{kind}.csv") for fold in range(1, 5)],
        ignore_index=True,
    )


def fold_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    per_fold = frame.groupby(["fold", *groups], as_index=False)[METRICS].mean()
    summary = per_fold.groupby(groups)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ablation = load("ablation")
    severity = load("severity")
    unimodal = load("unimodal")
    fold_summary(ablation, ["condition", "method"]).to_csv(
        OUTPUT / "ablation_summary.csv", index=False
    )
    fold_summary(severity, ["condition", "severity", "method"]).to_csv(
        OUTPUT / "severity_summary.csv", index=False
    )
    fold_summary(unimodal, ["modality"]).to_csv(
        OUTPUT / "unimodal_summary.csv", index=False
    )
    # Target-level severity data are retained for diagnosing sensor dependence.
    fold_summary(severity, ["condition", "severity", "target_modality", "method"]).to_csv(
        OUTPUT / "severity_by_modality.csv", index=False
    )
    print(f"Wrote extended summaries to {OUTPUT}")


if __name__ == "__main__":
    main()
