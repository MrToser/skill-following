# LOCKED: false
"""Render the skill-contract method overview.

The composition follows the supplied design reference, but all labels and
mathematical relations are aligned with the runtime used in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = ROOT / "contract_synchronized_overview"

INK = "#243447"
MUTED = "#5E6B78"
PURPLE = "#6542A6"
PURPLE_BG = "#F6F2FB"
BLUE = "#2E63B6"
BLUE_BG = "#F1F5FC"
GREEN = "#4E8B3D"
GREEN_BG = "#F2F8EF"
ORANGE = "#C78A20"
ORANGE_BG = "#FFF8E8"
LINE = "#CBD3DC"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "savefig.facecolor": "white",
    }
)


def box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str = LINE,
    face: str = "white",
    linewidth: float = 0.8,
    radius: float = 0.012,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 0.9,
    style: str = "-|>",
    connection: str = "arc3",
    dashed: bool = False,
    zorder: int = 3,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=linewidth,
        color=color,
        connectionstyle=connection,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)


def centered_text(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8.0,
    weight: str = "normal",
    color: str = INK,
    linespacing: float = 1.15,
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
        zorder=zorder,
    )


def draw_contract(ax: mpl.axes.Axes) -> None:
    box(ax, 0.012, 0.704, 0.976, 0.282, edge=LINE, face="#FCFCFD", linewidth=0.9)
    ax.text(0.030, 0.965, "Fixed expert procedure", fontsize=10.0, fontweight="bold", color=PURPLE, va="top")
    ax.text(0.250, 0.965, "one skill contract", fontsize=8.0, color=MUTED, va="top")

    box(ax, 0.026, 0.752, 0.194, 0.150, edge=PURPLE, face=PURPLE_BG)
    centered_text(ax, 0.123, 0.860, "Visible skill instructions  $z_j$", size=9.2, weight="bold", color=PURPLE)
    centered_text(ax, 0.123, 0.804, "shown to the executor", size=7.2, color=MUTED)

    box(ax, 0.242, 0.752, 0.332, 0.150, edge=PURPLE, face="white")
    ax.text(0.256, 0.895, "Contract phases  $\\mathcal{Q}_j$", fontsize=8.8, fontweight="bold", color=PURPLE, va="top")
    phases = ["Load", "Plan", "Act", "Use\nevidence", "Answer"]
    left = 0.253
    width = 0.050
    gap = 0.011
    for index, phase in enumerate(phases):
        x = left + index * (width + gap)
        box(ax, x, 0.792, width, 0.048, edge="#9A82C4", face=PURPLE_BG, radius=0.008)
        centered_text(ax, x + width / 2, 0.816, phase, size=7.0, weight="bold", color="#493276")
    ax.text(0.256, 0.756, "runtime-internal routing", fontsize=7.2, color=MUTED, va="bottom")

    box(ax, 0.594, 0.752, 0.380, 0.150, edge=GREEN, face="white")
    ax.text(0.608, 0.895, "Milestones  $\\mathcal{M}_j$", fontsize=8.8, fontweight="bold", color=GREEN, va="top")
    marks = ["$m_1$", "$m_2$", "$\\cdots$", "$m_k$", "Complete"]
    centers = [0.635, 0.704, 0.772, 0.840, 0.931]
    widths = [0.050, 0.050, 0.042, 0.050, 0.070]
    for index, (label, center, width) in enumerate(zip(marks, centers, widths)):
        if label == "$\\cdots$":
            centered_text(ax, center, 0.816, label, size=9.0, color=GREEN)
        else:
            box(ax, center - width / 2, 0.792, width, 0.048, edge=GREEN, face=GREEN_BG, radius=0.008)
            centered_text(ax, center, 0.816, label, size=8.0, weight="bold", color=GREEN)
        if index < len(marks) - 1:
            start = center + width / 2
            next_center = centers[index + 1]
            next_width = widths[index + 1]
            arrow(ax, (start + 0.002, 0.816), (next_center - next_width / 2 - 0.003, 0.816), color=GREEN, linewidth=0.7)
    ax.text(0.608, 0.756, "credited on first verified completion", fontsize=7.2, color=MUTED, va="bottom")


def draw_rollout(ax: mpl.axes.Axes) -> None:
    box(ax, 0.012, 0.247, 0.688, 0.421, edge=BLUE, face=BLUE_BG, linewidth=1.0)
    ax.text(0.028, 0.641, "On-policy rollout", fontsize=10.0, fontweight="bold", color=BLUE, va="top")
    ax.text(0.235, 0.641, "the contract checks each sampled action", fontsize=8.0, color=MUTED, va="top")

    box(ax, 0.035, 0.397, 0.105, 0.142, edge=BLUE, face="white")
    centered_text(ax, 0.0875, 0.488, "Policy", size=8.8, weight="bold", color=BLUE)
    centered_text(ax, 0.0875, 0.449, "$\\pi_\\theta$", size=10.0, color=BLUE)

    box(ax, 0.176, 0.411, 0.090, 0.112, edge=BLUE, face="white")
    centered_text(ax, 0.221, 0.482, "Action", size=8.2, weight="bold", color=BLUE)
    centered_text(ax, 0.221, 0.445, "$a_t$", size=9.5, color=BLUE)

    box(ax, 0.306, 0.385, 0.120, 0.164, edge=PURPLE, face=PURPLE_BG)
    centered_text(ax, 0.366, 0.505, "Contract step", size=8.4, weight="bold", color=PURPLE)
    centered_text(ax, 0.366, 0.463, "$\\mathrm{Step}_j$", size=9.2, color=INK)
    centered_text(ax, 0.366, 0.418, "parse · verify · route", size=7.2, color=MUTED)

    box(ax, 0.469, 0.385, 0.174, 0.164, edge=LINE, face="white")
    centered_text(ax, 0.556, 0.522, "Runtime outputs", size=8.0, weight="bold", color=INK)
    box(ax, 0.482, 0.451, 0.148, 0.050, edge=GREEN, face=GREEN_BG, radius=0.008)
    centered_text(ax, 0.556, 0.476, "Observation  $o_{t+1}$", size=8.0, weight="bold", color=GREEN)
    box(ax, 0.482, 0.397, 0.148, 0.042, edge=ORANGE, face=ORANGE_BG, radius=0.008)
    centered_text(ax, 0.556, 0.418, "Contract-state feedback", size=6.6, weight="bold", color=ORANGE)

    arrow(ax, (0.141, 0.468), (0.173, 0.468), color=BLUE)
    arrow(ax, (0.268, 0.468), (0.303, 0.468), color=BLUE)
    arrow(ax, (0.429, 0.468), (0.466, 0.468), color=PURPLE)
    arrow(ax, (0.645, 0.468), (0.716, 0.468), color=GREEN, linewidth=1.4)

    arrow(
        ax,
        (0.556, 0.394),
        (0.221, 0.402),
        color=ORANGE,
        connection="arc3,rad=-0.24",
        linewidth=0.9,
        dashed=True,
    )
    centered_text(ax, 0.389, 0.278, "after a nonterminal transition: append contract-state feedback", size=7.0, color=ORANGE)
    centered_text(ax, 0.344, 0.578, "$q_t \\rightarrow q_{t+1}$", size=8.5, color=PURPLE)
    centered_text(ax, 0.405, 0.578, "$\\Delta\\mathcal{M}_t$", size=8.5, color=GREEN)


def draw_credit(ax: mpl.axes.Axes) -> None:
    box(ax, 0.720, 0.247, 0.268, 0.421, edge=GREEN, face=GREEN_BG, linewidth=1.0)
    ax.text(0.737, 0.641, "Post-rollout credit", fontsize=10.0, fontweight="bold", color=GREEN, va="top")
    ax.text(0.737, 0.603, "verified progress yields distinct returns", fontsize=7.4, color=MUTED, va="top")

    box(ax, 0.752, 0.500, 0.204, 0.050, edge=GREEN, face="white", radius=0.008)
    centered_text(ax, 0.854, 0.525, "first-completed  $\\{\\Delta\\mathcal{M}_t\\}$", size=8.0, weight="bold", color=GREEN)
    arrow(ax, (0.854, 0.495), (0.854, 0.484), color=GREEN)

    box(ax, 0.745, 0.400, 0.218, 0.078, edge=GREEN, face="white", radius=0.008)
    centered_text(
        ax,
        0.854,
        0.439,
        "R = outcome +\nverified progress credit",
        size=7.4,
        color=INK,
    )
    arrow(ax, (0.854, 0.395), (0.854, 0.369), color=GREEN)

    box(ax, 0.776, 0.293, 0.156, 0.070, edge=GREEN, face="white", radius=0.008)
    centered_text(ax, 0.854, 0.347, "group-normalized  $\\widehat{A}_i$", size=9.0, weight="bold", color=GREEN)
    centered_text(ax, 0.854, 0.307, "trajectory-level GRPO", size=7.0, color=MUTED)


def draw_synchronized_outputs(ax: mpl.axes.Axes) -> None:
    box(ax, 0.012, 0.018, 0.976, 0.185, edge=PURPLE, face="#FCFBFE", linewidth=0.9)
    ax.text(0.030, 0.180, "One contract state, asymmetric learning roles", fontsize=9.0, fontweight="bold", color=PURPLE, va="top")

    box(ax, 0.040, 0.052, 0.225, 0.082, edge=PURPLE, face=PURPLE_BG, radius=0.008)
    centered_text(ax, 0.1525, 0.105, "Shared state  $q_t$", size=8.5, weight="bold", color=PURPLE)
    centered_text(ax, 0.1525, 0.074, "evaluated by  $\\mathrm{Step}_j$", size=7.2, color=MUTED)

    box(ax, 0.365, 0.052, 0.255, 0.082, edge=GREEN, face=GREEN_BG, radius=0.008)
    centered_text(ax, 0.4925, 0.105, "Verified progress", size=8.5, weight="bold", color=GREEN)
    centered_text(ax, 0.4925, 0.074, "$\\Delta\\mathcal{M}_t \\rightarrow$ trajectory credit", size=7.5, color=GREEN)

    box(ax, 0.710, 0.052, 0.248, 0.082, edge=ORANGE, face=ORANGE_BG, radius=0.008)
    centered_text(ax, 0.834, 0.105, "Contract-state feedback", size=8.0, weight="bold", color=ORANGE)
    centered_text(ax, 0.834, 0.074, "accepted: guidance · rejected: correction", size=7.1, color=ORANGE)

    arrow(ax, (0.268, 0.105), (0.360, 0.105), color=GREEN, linewidth=1.4)
    ax.plot([0.1525, 0.1525, 0.660], [0.050, 0.031, 0.031], color=ORANGE, linewidth=0.9, linestyle="--", zorder=3)
    arrow(ax, (0.660, 0.031), (0.705, 0.075), color=ORANGE, linewidth=0.9, dashed=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.20, 4.15))
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.01, top=0.995)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_contract(ax)
    draw_rollout(ax)
    draw_credit(ax)
    draw_synchronized_outputs(ax)

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
