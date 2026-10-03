"""Create the main robustness figure from locked-fold summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_DIR = Path("results/locked_summary")
PAPER_FIGURE_DIR = Path("paper/figures")
OVERLEAF_FIGURE_DIR = Path("paper/overleaf/figures")
ORDER = ["clean", "partial_dropout", "gaussian", "bias", "replay"]
LABELS = ["Clean", "Partial\ndropout", "Gaussian\nnoise", "Bias", "Replay"]


def main() -> None:
    conditions = pd.read_csv(SUMMARY_DIR / "condition_summary.csv")
    effects = pd.read_csv(SUMMARY_DIR / "paired_user_effects.csv")
    effects = effects[effects["metric"].eq("f1")].set_index("condition").loc[ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    x = np.arange(len(ORDER))
    width = 0.36
    colors = {"uniform_late_fusion": "#777777", "reliability_fusion": "#176B87"}
    names = {"uniform_late_fusion": "Uniform", "reliability_fusion": "Trust-aware"}

    for offset, method in zip((-width / 2, width / 2), colors):
        subset = conditions[conditions["method"].eq(method)].set_index("condition").loc[ORDER]
        axes[0].bar(
            x + offset,
            subset["f1_mean"],
            width,
            yerr=subset["f1_std"],
            capsize=2,
            color=colors[method],
            label=names[method],
            linewidth=0,
        )
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.35, 0.55)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].set_title("(a) Locked folds 1–4")

    mean = effects["mean_delta"].to_numpy()
    low = effects["ci95_low"].to_numpy()
    high = effects["ci95_high"].to_numpy()
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].errorbar(
        mean,
        x,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color="#176B87",
        capsize=3,
        markersize=4,
    )
    axes[1].set_yticks(x, LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Participant-paired Δ macro-F1")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_title("(b) 95% bootstrap CI")

    fig.savefig(SUMMARY_DIR / "robustness_main.pdf", bbox_inches="tight")
    fig.savefig(SUMMARY_DIR / "robustness_main.png", dpi=220, bbox_inches="tight")
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURE_DIR / "robustness_main.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURE_DIR / "robustness_main.png", dpi=220, bbox_inches="tight")

    # Main-paper panel: aggregate performance only.
    fig_main, axis = plt.subplots(figsize=(6.2, 2.55), constrained_layout=True)
    for offset, method in zip((-width / 2, width / 2), colors):
        subset = conditions[conditions["method"].eq(method)].set_index("condition").loc[ORDER]
        axis.bar(
            x + offset,
            subset["f1_mean"],
            width,
            yerr=subset["f1_std"],
            capsize=2,
            color=colors[method],
            label=names[method],
            linewidth=0,
        )
    axis.set_xticks(x, LABELS)
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0.35, 0.55)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    for extension in ("pdf", "png"):
        fig_main.savefig(
            PAPER_FIGURE_DIR / f"robustness_conditions.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig_main)

    # Compact single-column version for more predictable IEEE float placement.
    fig_column, axis = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    for offset, method in zip((-width / 2, width / 2), colors):
        subset = conditions[conditions["method"].eq(method)].set_index("condition").loc[ORDER]
        axis.bar(
            x + offset,
            subset["f1_mean"],
            width,
            yerr=subset["f1_std"],
            capsize=1.5,
            color=colors[method],
            label=names[method],
            linewidth=0,
        )
    axis.set_xticks(x, ["Clean", "Partial\nloss", "Gaussian", "Bias", "Replay"])
    axis.tick_params(axis="both", labelsize=7)
    axis.set_ylabel("Macro-F1", fontsize=8)
    axis.set_ylim(0.35, 0.55)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="upper right", fontsize=7)
    OVERLEAF_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig_column.savefig(
            OVERLEAF_FIGURE_DIR / f"robustness_conditions_singlecol.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig_column)

    # Presentation appendix panel: participant-paired confidence intervals.
    fig_ci, axis = plt.subplots(figsize=(5.6, 2.55), constrained_layout=True)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.errorbar(
        mean,
        x,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color="#176B87",
        capsize=3,
        markersize=4,
    )
    axis.set_yticks(x, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Participant-paired Δ macro-F1")
    axis.grid(axis="x", alpha=0.25)
    for extension in ("pdf", "png"):
        fig_ci.savefig(
            PAPER_FIGURE_DIR / f"participant_ci.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig_ci)
    print(SUMMARY_DIR / "robustness_main.png")


if __name__ == "__main__":
    main()
