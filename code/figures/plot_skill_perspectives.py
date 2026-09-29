# LOCKED: false
"""Render a compact positioning figure for grounded skill-following."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = ROOT / "skill_perspectives"

INK = "#253245"
MUTED = "#66717E"
GRAY = "#64717F"
GRAY_BG = "#F5F6F7"
BLUE = "#2E63B6"
GREEN = "#4E8B3D"
ORANGE = "#C78A20"
PURPLE = "#6743A5"
PURPLE_BG = "#F6F2FB"


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


def rounded_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str,
    face: str,
    linewidth: float = 0.8,
    radius: float = 0.014,
) -> None:
    """Data: normalized geometry and colors. Algorithm: draw one rounded box."""
    assert width > 0 and height > 0
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
        )
    )


def label(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.6,
    weight: str = "normal",
    color: str = INK,
    align: str = "center",
) -> None:
    """Data: label and anchor. Algorithm: place readable publication text."""
    ax.text(
        x,
        y,
        text,
        ha=align,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.18,
    )


def comparison_row(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    heading: str,
    content: str,
    *,
    color: str,
    line_colors: tuple[str, str] | None = None,
) -> None:
    """Data: one comparison attribute. Algorithm: pair a heading with its value."""
    label(ax, x + 0.151, y + 0.119, heading, size=7.2, weight="bold", color=MUTED)
    rounded_box(ax, x + 0.025, y, 0.252, 0.092, edge=color, face="white", radius=0.008)
    if line_colors is None:
        label(ax, x + 0.151, y + 0.046, content, size=8.0)
    else:
        first, second = content.split("\n", maxsplit=1)
        label(ax, x + 0.151, y + 0.063, first, size=7.0, color=line_colors[0])
        label(ax, x + 0.151, y + 0.027, second, size=7.0, color=line_colors[1])


def draw_view(
    ax: mpl.axes.Axes,
    x: float,
    *,
    panel: str,
    title: str,
    subtitle: str,
    supplied_text: str,
    fixed_text: str,
    optimized_text: str,
    interface_text: str,
    color: str,
    background: str,
    emphasized: bool = False,
    interface_colors: tuple[str, str] | None = None,
) -> None:
    """Data: one research emphasis. Algorithm: render four shared dimensions."""
    rounded_box(
        ax,
        x,
        0.025,
        0.302,
        0.950,
        edge=color,
        face=background,
        linewidth=1.3 if emphasized else 0.85,
        radius=0.018,
    )
    label(ax, x + 0.030, 0.916, panel, size=9.4, weight="bold", color=color)
    label(ax, x + 0.151, 0.916, title, size=9.4, weight="bold", color=color)
    label(ax, x + 0.151, 0.858, subtitle, size=8.0, color=MUTED)
    comparison_row(ax, x, 0.665, "Supplied object", supplied_text, color=PURPLE)
    comparison_row(ax, x, 0.485, "Fixed object", fixed_text, color=PURPLE)
    comparison_row(ax, x, 0.305, "Optimized object", optimized_text, color=BLUE)
    comparison_row(
        ax,
        x,
        0.125,
        "Learning interface",
        interface_text,
        color=GREEN if interface_colors else GRAY,
        line_colors=interface_colors,
    )


def main() -> None:
    """Data: three complementary research emphases. Algorithm: export one schematic."""
    fig, ax = plt.subplots(figsize=(7.18, 3.00))
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.02, top=0.995)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_view(
        ax,
        0.011,
        panel="(a)",
        title="Instruction following",
        subtitle="Representative emphasis",
        supplied_text="Current instruction",
        fixed_text="Requested task and constraints",
        optimized_text="Response or executor policy",
        interface_text="Response or action compliance",
        color=GRAY,
        background=GRAY_BG,
    )
    draw_view(
        ax,
        0.349,
        panel="(b)",
        title="Skill lifecycle",
        subtitle="Representative emphasis",
        supplied_text="Task goal and skill resources",
        fixed_text="Task objective",
        optimized_text="Skill resource and/or executor",
        interface_text="Discovery, retrieval, adaptation,\nor internalization",
        color=GRAY,
        background=GRAY_BG,
    )
    draw_view(
        ax,
        0.687,
        panel="(c)",
        title="Grounded skill-following",
        subtitle="Focus of this work",
        supplied_text="Visible skill instructions  $z_j$",
        fixed_text="Skill contract  $\\mathcal{C}_j$",
        optimized_text="Executor policy",
        interface_text="Verified progress credit after rollout\nContract-state feedback during rollout",
        color=PURPLE,
        background=PURPLE_BG,
        emphasized=True,
        interface_colors=(GREEN, ORANGE),
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
