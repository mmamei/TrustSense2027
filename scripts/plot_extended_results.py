"""Create publication-ready severity and ablation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SUMMARY = Path("results/extended_summary")
FIGURES = Path("paper/figures")
NAMES = {
    "partial_dropout": "Partial dropout",
    "gaussian": "Gaussian noise",
    "bias": "Additive bias",
    "replay": "Replay",
}
METHOD_NAMES = {
    "uniform": "Uniform",
    "completeness_only": "Completeness",
    "ood_only": "OOD",
    "agreement_only": "Agreement",
    "full_reliability": "Full",
}
MODALITY_NAMES = {
    "phone_acc": "Phone acc.", "phone_gyro": "Phone gyro.",
    "phone_magnet": "Magnetometer", "watch_acc": "Watch acc.",
    "location": "Location", "audio": "Audio",
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def severity_figure() -> None:
    data = pd.read_csv(SUMMARY / "severity_summary.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.05), sharey=True)
    colors = {"uniform": "#777777", "full_reliability": "#0072B2"}
    markers = {"uniform": "o", "full_reliability": "s"}
    for axis, condition in zip(axes, NAMES):
        subset = data[data["condition"].eq(condition)]
        for method in ("uniform", "full_reliability"):
            rows = subset[subset["method"].eq(method)].sort_values("severity")
            axis.errorbar(
                rows["severity"], rows["f1_mean"], yerr=rows["f1_std"],
                color=colors[method], marker=markers[method], linewidth=1.8,
                capsize=2.5, label=METHOD_NAMES[method],
            )
        axis.set_title(NAMES[condition])
        axis.set_xlabel("Corruption severity")
        axis.set_ylim(0.30, 0.51)
        axis.tick_params(axis="both", labelsize=7)
        axis.title.set_fontsize(8)
        axis.xaxis.label.set_size(7.5)
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.8)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"severity_curves.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ablation_figure() -> None:
    ablation = pd.read_csv(SUMMARY / "ablation_summary.csv")
    unimodal = pd.read_csv(SUMMARY / "unimodal_summary.csv")
    order = ["uniform", "completeness_only", "ood_only", "agreement_only", "full_reliability"]
    conditions = ["clean", "partial_dropout", "gaussian", "bias", "replay"]
    pivot = ablation.pivot(index="method", columns="condition", values="f1_mean").loc[order, conditions]
    delta = pivot.subtract(pivot.loc["uniform"], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), gridspec_kw={"width_ratios": [1.45, 1]})
    sns.heatmap(
        delta, ax=axes[0], cmap="RdBu", center=0, vmin=-0.03, vmax=0.03,
        annot=True, fmt="+.3f", linewidths=0.5,
        annot_kws={"fontsize": 7.2},
        cbar_kws={"label": r"$\Delta$ macro-F1 vs. uniform", "shrink": 0.85},
    )
    axes[0].set_title("(a) Reliability ablation")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_xticklabels(["Clean", "Dropout", "Gaussian", "Bias", "Replay"], rotation=30, ha="right")
    axes[0].set_yticklabels([METHOD_NAMES[item] for item in order], rotation=0)

    rows = unimodal.sort_values("f1_mean", ascending=True)
    axes[1].barh(
        [MODALITY_NAMES[item] for item in rows["modality"]], rows["f1_mean"],
        xerr=rows["f1_std"], color="#9ecae1", edgecolor="#2171b5", capsize=2,
    )
    clean = pivot["clean"]
    axes[1].axvline(clean["uniform"], color="#777777", linestyle="--", linewidth=1.5, label="Uniform fusion")
    axes[1].axvline(clean["full_reliability"], color="#0072B2", linewidth=1.8, label="Full fusion")
    axes[1].set_xlim(0.25, 0.52)
    axes[1].set_xlabel("Clean macro-F1")
    axes[1].set_title("(b) Unimodal baselines")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"ablation_unimodal.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Main-paper panel: unimodal references only.
    fig_uni, axis = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    axis.barh(
        [MODALITY_NAMES[item] for item in rows["modality"]], rows["f1_mean"],
        xerr=rows["f1_std"], color="#9ecae1", edgecolor="#2171b5", capsize=2,
    )
    axis.axvline(clean["uniform"], color="#777777", linestyle="--", linewidth=1.5,
                 label="Uniform fusion")
    axis.axvline(clean["full_reliability"], color="#0072B2", linewidth=1.8,
                 label="Full fusion")
    axis.set_xlim(0.25, 0.52)
    axis.set_xlabel("Clean macro-F1")
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    for extension in ("pdf", "png"):
        fig_uni.savefig(FIGURES / f"unimodal_baselines.{extension}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig_uni)

    # Presentation appendix panel: component ablation only.
    fig_abl, axis = plt.subplots(figsize=(6.0, 2.9), constrained_layout=True)
    sns.heatmap(
        delta, ax=axis, cmap="RdBu", center=0, vmin=-0.03, vmax=0.03,
        annot=True, fmt="+.3f", linewidths=0.5,
        annot_kws={"fontsize": 8},
        cbar_kws={"label": r"$\Delta$ macro-F1 vs. uniform", "shrink": 0.85},
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticklabels(["Clean", "Dropout", "Gaussian", "Bias", "Replay"],
                         rotation=20, ha="right")
    axis.set_yticklabels([METHOD_NAMES[item] for item in order], rotation=0)
    for extension in ("pdf", "png"):
        fig_abl.savefig(FIGURES / f"ablation_components.{extension}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig_abl)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    severity_figure()
    ablation_figure()
    print(FIGURES / "severity_curves.png")
    print(FIGURES / "ablation_unimodal.png")


if __name__ == "__main__":
    main()
