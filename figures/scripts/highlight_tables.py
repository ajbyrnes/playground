"""Annotated copies of extracted paper figures, for the slides:
- Ozaki et al.'s Tables 3 and 4 with the Ozaki scheme's columns (M3-M5,
  k = 2, 3, 4 slices) boxed, for the Evaluation: Results slide;
- Ootomo et al.'s Fig. 9 with its three rows labeled.

Reads the extracted tables in figures/papers/ozaki/ (see paper_figures.py)
and writes *_accmul.png / *_labeled.png next to them. Those stay out of git, like the papers.

Run: .venv/bin/python figures/scripts/highlight_tables.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PAPERS = Path(__file__).parent.parent / "papers"
DIR = PAPERS / "ozaki"
COLOR = "#d9480f"  # the slides' magnitude range color

# (left, top, right, bottom) in pixels: from the gap after M2's column to the
# gap before M6's, between the top and bottom rules (measured from the ink)
BOXES = {
    "table03": (468, 17, 1030, 191),
    "table04": (318, 17, 685, 192),
}

# Ootomo et al. Fig. 9: each row's vertical center, in pixels, and its label
ROW_LABELS = [(258, "Throughput"), (505, "Power\nefficiency"), (752, "Average\npower\nconsumption")]


def label_rows(src, out, rows, margin=260, fontsize=12):
    """src with a white margin on the left holding a label per row."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    dpi = 150
    fig = plt.figure(figsize=((w + margin) / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im, extent=(margin, w + margin, h, 0))
    ax.set_xlim(0, w + margin)
    ax.set_ylim(h, 0)
    ax.axis("off")
    for y, text in rows:
        ax.text(margin * 0.5, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", linespacing=1.2)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    src = PAPERS / "ootomo" / "fig09.png"
    if src.exists():
        label_rows(src, PAPERS / "ootomo" / "fig09_labeled.png", ROW_LABELS)

    for name, box in BOXES.items():
        src = DIR / f"{name}.png"
        if not src.exists():
            print("missing", src, "(run paper_figures.py first)")
            continue
        im = Image.open(src).convert("RGB")
        ImageDraw.Draw(im).rounded_rectangle(box, radius=8, outline=COLOR, width=4)
        out = DIR / f"{name}_accmul.png"
        im.save(out)
        print("wrote", out)
