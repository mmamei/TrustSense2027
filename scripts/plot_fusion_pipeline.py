"""Draw the reliability-aware late-fusion pipeline used in the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("paper/overleaf/figures")


def box(ax, x, y, w, h, text, face, fontsize=7.0, edge="#34495e"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=face, edgecolor=edge, linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#34495e"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9,
        linewidth=1.0, color=color, shrinkA=1, shrinkB=1,
    ))


def elbow_arrow(ax, points, color="#b36b1e"):
    """Draw an orthogonal connector whose arrowhead ends at the last point."""
    for first, second in zip(points[:-2], points[1:-1]):
        ax.plot([first[0], second[0]], [first[1], second[1]],
                color=color, linewidth=1.0)
    arrow(ax, points[-2], points[-1], color=color)


def main():
    fig, ax = plt.subplots(figsize=(7.15, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#dceff5"
    green = "#dcefdc"
    orange = "#f8e6cf"
    purple = "#e9e1f4"
    grey = "#eef1f3"
    y, h = 0.40, 0.40

    box(ax, 0.020, y, 0.105, h,
        "Raw sensor\nstream\nexample: phone\naccelerometer\n$[0.12,-0.31,\\ldots]$",
        grey, fontsize=5.3)
    box(ax, 0.155, y, 0.115, h,
        "Per-modality\nfeatures\n$\\mathbf{x}_m=[0.42,\\ldots]$\n$80/100$ coordinates\npresent",
        blue, fontsize=5.3)
    box(ax, 0.300, y, 0.150, h,
        "Base\nclassifiers\n$\\mathbf{p}_m=f_m(\\mathbf{x}_m)$\n"
        "$p_{\\mathrm{run}}=0.72$\n$p_{\\mathrm{outside}}=0.18$",
        green, fontsize=5.3)
    box(ax, 0.485, y, 0.120, h,
        "Reliability-aware\nweights\n$w_{m,\\mathrm{run}}=0.66$\n"
        "$w_{m,\\mathrm{outside}}=0.55$",
        orange, fontsize=5.3)
    box(ax, 0.640, y, 0.140, h,
        "Weighted\nlate fusion\n$\\hat p_{\\mathrm{run}}=0.79$\n"
        "$\\hat p_{\\mathrm{outside}}=0.24$",
        purple, fontsize=5.3)
    box(ax, 0.820, y, 0.160, h,
        "$L$ binary\noutputs\n$\\hat y_{\\mathrm{run}}=1$\n"
        "$\\hat y_{\\mathrm{outside}}=0$",
        purple, fontsize=5.3)

    for start, end in [
        ((0.128, 0.60), (0.147, 0.60)),
        ((0.278, 0.60), (0.292, 0.60)),
        ((0.453, 0.60), (0.477, 0.60)),
        ((0.608, 0.60), (0.632, 0.60)),
        ((0.783, 0.60), (0.812, 0.60)),
    ]:
        arrow(ax, start, end)

    box(ax, 0.150, 0.070, 0.510, 0.220,
        "Reliability evidence for this example\n"
        "$c_m=0.80$ (completeness)   |   $o_m=0.91$ (familiarity)\n"
        "$a_{m,\\mathrm{run}}=0.76$ (cross-modal agreement)",
        orange, fontsize=5.9)

    arrow(ax, (0.212, 0.395), (0.212, 0.300), color="#b36b1e")
    arrow(ax, (0.375, 0.395), (0.375, 0.300), color="#b36b1e")
    arrow(ax, (0.545, 0.300), (0.545, 0.395), color="#b36b1e")

    ax.text(0.252, 0.835, "upstream sensing and prediction", ha="center",
            fontsize=7.2, color="#566573")
    ax.text(0.750, 0.835, "proposed reliability-aware fusion", ha="center",
            fontsize=7.2, color="#9a5a18")
    ax.text(0.5, 0.965, "Illustrative example: one sensing window (simulated values)",
            ha="center", va="center", fontsize=7.7, fontweight="bold",
            color="#2c3e50")
    ax.plot([0.470, 0.470], [0.36, 0.81], linestyle="--", linewidth=0.8,
            color="#7f8c8d")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fusion_pipeline.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
