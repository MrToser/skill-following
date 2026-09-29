# LOCKED: false
"""Plot the Qwen3.5-4B Task Outcome rows reported in the paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "qwen35_main_results.csv"
OUTPUT = ROOT / "assets" / "qwen35_task_outcome"
ORDER = ("Direct", "Prompt", "Recipe", "Ours-S", "Ours-J")
COLORS = {
    "Direct": "#AEB5BC",
    "Prompt": "#D4D8DC",
    "Recipe": "#758692",
    "Ours-S": "#6655A7",
    "Ours-J": "#D9874B",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    }
)


# Data: the five Qwen3.5-4B rows from the paper table.
# Algorithm: read the source CSV and assert that every expected setting appears once.
def load_rows() -> dict[str, dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(ORDER)
    by_setting = {row["setting"]: row for row in rows}
    assert set(by_setting) == set(ORDER)
    for row in rows:
        for field in ("math_task_outcome", "search_task_outcome"):
            value = float(row[field])
            assert 0 <= value <= 100
    return by_setting


# Data: the two domain averages for all five settings.
# Algorithm: draw aligned horizontal bars on a common 0-100 scale with direct value labels.
def make_figure(rows: dict[str, dict[str, str]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
    for axis, domain in zip(axes, ("math", "search")):
        for position, setting in enumerate(ORDER):
            value = float(rows[setting][f"{domain}_task_outcome"])
            axis.barh(position, value, height=0.68, color=COLORS[setting], edgecolor="none")
            axis.text(value + 1.4, position, f"{value:.2f}", va="center", ha="left", fontsize=8)
        axis.set_yticks(range(len(ORDER)), ORDER)
        axis.set_xlim(0, 100)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.set_xlabel("Task Outcome (%)", fontsize=8)
        axis.set_title(domain.capitalize(), loc="left", fontsize=10, fontweight="bold")
        axis.set_axisbelow(True)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#AEB5BC")
        axis.tick_params(axis="both", length=0, pad=5)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.17, top=0.88, wspace=0.25)
    return fig


def main() -> None:
    rows = load_rows()
    fig = make_figure(rows)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
