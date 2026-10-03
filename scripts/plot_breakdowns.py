"""Generate label, modality, and participant-level robustness breakdowns."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("results/full_experiment")
OUTPUT = Path("paper/figures")
METHODS = ("uniform_late_fusion", "reliability_fusion")
CONDITIONS = ["partial_dropout", "gaussian", "bias", "replay"]
CONDITION_LABELS = ["Partial dropout", "Gaussian noise", "Bias", "Replay"]
LABEL_NAMES = {
    "label:SITTING": "Sitting",
    "label:LYING_DOWN": "Lying down",
    "label:OR_standing": "Standing",
    "label:FIX_walking": "Walking",
    "label:BICYCLING": "Bicycling",
    "label:IN_A_CAR": "In a car",
    "label:TALKING": "Talking",
    "label:OR_outside": "Outside",
}
MODALITY_NAMES = {
    "phone_acc": "Phone acc.",
    "phone_gyro": "Phone gyro.",
    "phone_magnet": "Magnetometer",
    "watch_acc": "Watch acc.",
    "location": "Location",
    "audio": "Audio",
}


def load(suffix: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(INPUT / f"fold_[1-4]_{suffix}.csv")))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame[frame["method"].isin(METHODS)].copy()


def method_delta(
    frame: pd.DataFrame, index: list[str], metric: str = "f1"
) -> pd.Series:
    grouped = frame.groupby([*index, "method"])[metric].mean().unstack("method")
    return grouped["reliability_fusion"] - grouped["uniform_late_fusion"]


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > 0.025 else "#222222"
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7, color=color)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = load("metrics")
    users = load("user_metrics")

    corrupted = metrics[metrics["condition"].isin(CONDITIONS)]
    label_delta = method_delta(corrupted, ["condition", "label"]).unstack("condition")
    label_delta = label_delta.loc[list(LABEL_NAMES), CONDITIONS]
    modality_delta = method_delta(
        corrupted, ["condition", "target_modality"]
    ).unstack("condition")
    modality_delta = modality_delta.loc[list(MODALITY_NAMES), CONDITIONS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    vmax = max(abs(label_delta.to_numpy()).max(), abs(modality_delta.to_numpy()).max())
    for axis, table, ylabels, title in (
        (axes[0], label_delta, list(LABEL_NAMES.values()), "(a) By context label"),
        (axes[1], modality_delta, list(MODALITY_NAMES.values()), "(b) By corrupted modality"),
    ):
        image = axis.imshow(table.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(np.arange(len(CONDITIONS)), CONDITION_LABELS, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(ylabels)), ylabels)
        axis.set_title(title)
        annotate_heatmap(axis, table.to_numpy())
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Δ macro-F1 (trust-aware − uniform)")
    fig.savefig(OUTPUT / "breakdown_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "breakdown_heatmaps.png", dpi=220, bbox_inches="tight")

    # IEEE single-column variant: stack the two panels so that the final result
    # can share a page with discussion and references instead of forcing a
    # mostly empty full-width float page.
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.0), constrained_layout=True)
    for axis, table, ylabels, title in (
        (axes[0], label_delta, list(LABEL_NAMES.values()), "(a) By context label"),
        (axes[1], modality_delta, list(MODALITY_NAMES.values()), "(b) By corrupted modality"),
    ):
        image = axis.imshow(table.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(np.arange(len(CONDITIONS)), CONDITION_LABELS, rotation=20, ha="right")
        axis.set_yticks(np.arange(len(ylabels)), ylabels)
        axis.set_title(title, fontsize=9)
        axis.tick_params(labelsize=7)
        annotate_heatmap(axis, table.to_numpy())
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", shrink=0.82, pad=0.01)
    colorbar.set_label("Δ macro-F1 (trust-aware − uniform)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(OUTPUT / "breakdown_heatmaps_singlecol.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "breakdown_heatmaps_singlecol.png", dpi=220, bbox_inches="tight")

    per_user = (
        users.groupby(["fold", "user_id", "condition", "method"], as_index=False)["f1"]
        .mean()
    )
    pivot = per_user.pivot(
        index=["fold", "user_id", "condition"], columns="method", values="f1"
    ).dropna()
    pivot["delta"] = pivot["reliability_fusion"] - pivot["uniform_late_fusion"]
    data = [pivot.xs(condition, level="condition")["delta"].to_numpy() for condition in ["clean", *CONDITIONS]]

    fig, axis = plt.subplots(figsize=(5.8, 2.8), constrained_layout=True)
    boxes = axis.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
    for box in boxes["boxes"]:
        box.set_facecolor("#8CC4D5")
        box.set_edgecolor("#176B87")
    rng = np.random.default_rng(2027)
    for index, values in enumerate(data, start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        axis.scatter(index + jitter, values, s=9, alpha=0.35, color="#174A5B", linewidth=0)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xticks(np.arange(1, 6), ["Clean", *CONDITION_LABELS], rotation=18, ha="right")
    axis.set_ylabel("Participant-paired Δ macro-F1")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT / "participant_effects.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "participant_effects.png", dpi=220, bbox_inches="tight")
    print(OUTPUT / "breakdown_heatmaps.png")
    print(OUTPUT / "participant_effects.png")


if __name__ == "__main__":
    main()
