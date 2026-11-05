"""SVG figures for the Applications section (Brower et al. 2026, DMRG).

sector_blocks  a D x D matrix split by quantum numbers into sectors: only some
               row/column sector pairs hold a dense block, and each DGEMM works
               on one block (m x n), never on the whole D x D matrix.

The sector sizes are made up: none of the DMRG papers report them.

Run: .venv/bin/python figures/scripts/applications.py   → writes figures/static/slides/sector_blocks.svg
"""
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figures as F

OUT = Path(__file__).parent.parent / "static" / "slides"
plt.rcParams.update({"font.size": 11, "svg.fonttype": "path"})

BLOCK = "#cfe0f5"
EDGE = "#1c4e8a"
PICK = "#d9480f"

SECTORS = [3, 5, 7, 6, 4, 3]                  # rows and columns, same split
FILLED = {(i, i) for i in range(6)} | {(i, i + 1) for i in range(5)} | {(i + 1, i) for i in range(5)}
PICKED = (2, 3)                               # the block one DGEMM works on


def figure_sector_blocks():
    D = sum(SECTORS)
    starts = [0, *accumulate(SECTORS)]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(Rectangle((0, 0), D, D, facecolor="#f4f4f4", edgecolor="#222", linewidth=1.2))
    for k in starts[1:-1]:
        ax.plot([k, k], [0, D], color="#bbb", linewidth=0.6, dashes=(2, 2))
        ax.plot([0, D], [k, k], color="#bbb", linewidth=0.6, dashes=(2, 2))
    for i, j in FILLED:
        picked = (i, j) == PICKED
        ax.add_patch(Rectangle((starts[j], starts[i]), SECTORS[j], SECTORS[i],
                               facecolor=PICK if picked else BLOCK, alpha=0.85 if picked else 1,
                               edgecolor=PICK if picked else EDGE, linewidth=1.6 if picked else 0.8))
    i, j = PICKED
    x0, y0, w, h = starts[j], starts[i], SECTORS[j], SECTORS[i]
    ax.text(x0 + w / 2, y0 + h / 2, rf"$m \times n$", ha="center", va="center", fontsize=18, color="white")
    # D along the top and the left
    F.span_arrow(ax, 0, D, -1.2, "$D$", fontsize=26)
    ax.annotate("", (-1.2, D), xytext=(-1.2, 0),
                arrowprops=dict(arrowstyle="<->", linewidth=0.8, color="#222", shrinkA=0, shrinkB=0))
    ax.text(-1.7, D / 2, "$D$", ha="right", va="center", fontsize=26)
    ax.set_xlim(-4.2, D + 0.4)
    ax.set_ylim(D + 0.4, -4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig = figure_sector_blocks()
    fig.savefig(OUT / "sector_blocks.svg", bbox_inches="tight", pad_inches=0.05)
    print("wrote", OUT / "sector_blocks.svg")
