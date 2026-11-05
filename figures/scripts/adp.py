"""SVG figures for the ADP section (Schwarz et al. 2026): what ESC covers.

esc_matrix_1..3  A · B = C, built up in three steps for one entry c_11:
                 (1) the row of A and column of B behind it,
                 (2) the largest value in that row and in that column,
                 (3) the pair whose product is the largest term of c_11.
eight_bit_slices row 1 of A: (a) its bits, (b) the same bits cut into 8-bit
                 slices from the top, as many as it takes to hold them all.
esc_slices       the six values behind c_11 in mantissa space: 8-bit slices laid
                 over the top 53 + ESC bits, everything below dumped.

The slides handwave the sign: every slice is drawn as 8 mantissa bits (in ADP
the first slice also carries the sign; see figures.py's figure_slice_esc).

ESC follows the paper (Section 4), for one dot product x · y:
    ESC = exp(x_p) + exp(y_q) - exp(z_r),   z_i = x_i y_i,
and the ESC of A · B is the largest over all its dot products. (The paper
adds 1 for safety; the figures leave that out.)

Run: .venv/bin/python figures/scripts/adp.py   → writes figures/static/slides/esc_*.svg
"""
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import figures as F

OUT = Path(__file__).parent.parent / "static" / "slides"
plt.rcParams.update({"font.size": 11, "svg.fonttype": "path"})

LARGEST = "#1c4e8a"   # largest in the row of A / column of B (the deck's blue)
CONTRIB = "#d9480f"   # the pair behind the largest product term
BAND = "#e3edf9"      # the row and column behind c_11
DUMPED = F.RED
MUTED = "#9a9a9a"

# exponents: row 1 of A and column 1 of B carry the example, the rest stay near 2^0
#   x = row 1 of A:     2^0 (x_p),   2^-3 (x_r),  2^-30
#   y = column 1 of B:  2^-20,       2^-8 (y_r),  2^0 (y_q)
#   products:           2^-20,       2^-11 (z_r), 2^-30   ->  ESC = 0 + 0 - (-11) = 11
# so 53 + ESC = 64 bits = exactly 8 slices
EXP_A = [[0, -3, -30], [-1, 0, -2], [0, -2, -1]]
EXP_B = [[-20, -1, 0], [-8, 0, -2], [0, -2, -1]]
P, Q, R = 0, 2, 1          # x_p = a_11, y_q = b_31, z_r = a_12 b_21


def matrix(exps, seed):
    """Positive values with these exponents and random 52-bit mantissas
    (positive, so dropping bits is plain truncation)."""
    rng = random.Random(seed)
    return [[(1 + rng.getrandbits(52) / 2**52) * 2.0**e for e in row] for row in exps]


A, B = matrix(EXP_A, 3), matrix(EXP_B, 8)
C = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
x, y = A[0], [B[k][0] for k in range(3)]
ESC = F.esc_of(A, B)
assert ESC == F.exp_of(x[P]) + F.exp_of(y[Q]) - (F.exp_of(x[R]) + F.exp_of(y[R])) == 11
W = 53 + ESC


def entry(v, digits=2):
    e = F.exp_of(v)
    return rf"${v / 2.0**e:.{digits}f}\times 2^{{{e}}}$"


# ---- slide 1: which values ESC looks at ----

CW, CH = 2.0, 0.95     # cell size, data units
CW_C = 1.15            # C's cells: only c_11 is named, the rest are dots


def draw_matrix(ax, x0, M, name, band=None, circles=(), muted=True, cw=CW, labels=None):
    """A 3 x 3 matrix with brackets, its top-left corner at (x0, 0).
    band = ("row", i) or ("col", j) or ("cell", (i, j)); circles = [(i, j, color)]."""
    CW = cw
    w, h = 3 * CW, 3 * CH
    if band:
        kind, k = band
        if kind == "row":
            ax.add_patch(Rectangle((x0, k * CH), w, CH, facecolor=BAND, edgecolor="none"))
        elif kind == "col":
            ax.add_patch(Rectangle((x0 + k * CW, 0), CW, h, facecolor=BAND, edgecolor="none"))
        else:
            i, j = k
            ax.add_patch(Rectangle((x0 + j * CW, i * CH), CW, CH, facecolor=BAND, edgecolor="none"))

    def in_band(i, j):
        if not band:
            return True
        kind, k = band
        return (kind == "row" and i == k) or (kind == "col" and j == k) or (kind == "cell" and (i, j) == k)

    for i in range(3):
        for j in range(3):
            text = labels[i][j] if labels else entry(M[i][j], 1)
            ax.text(x0 + (j + 0.5) * CW, (i + 0.5) * CH, text, ha="center", va="center",
                    fontsize=13, color="#222" if in_band(i, j) or not muted else MUTED)
    for i, j, color in circles:
        ax.add_patch(Ellipse((x0 + (j + 0.5) * CW, (i + 0.5) * CH), CW * 0.98, CH * 0.92,
                             facecolor="none", edgecolor=color, linewidth=2.2))
    # brackets
    for side, sx in ((-1, x0 - 0.08), (1, x0 + w + 0.08)):
        ax.plot([sx - side * 0.18, sx, sx, sx - side * 0.18], [-0.08, -0.08, h + 0.08, h + 0.08],
                color="#222", linewidth=1.1, solid_capstyle="butt")
    ax.text(x0 + w / 2, -0.35, f"${name}$", ha="center", va="bottom", fontsize=16)


def figure_esc_matrix(step):
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    gap = 1.0
    xa = 0
    xb = xa + 3 * CW + gap
    xc = xb + 3 * CW + gap
    ax.text(xb - gap / 2, 1.5 * CH, r"$\cdot$", ha="center", va="center", fontsize=22)
    ax.text(xc - gap / 2, 1.5 * CH, "$=$", ha="center", va="center", fontsize=16)

    ca, cb = [], []
    if step >= 2:
        ca.append((0, P, LARGEST))
        cb.append((Q, 0, LARGEST))
    if step >= 3:
        ca.append((0, R, CONTRIB))
        cb.append((R, 0, CONTRIB))
    draw_matrix(ax, xa, A, "A", ("row", 0), ca)
    draw_matrix(ax, xb, B, "B", ("col", 0), cb)
    dots = [["$c_{11}$" if (i, j) == (0, 0) else r"$\bullet$" for j in range(3)] for i in range(3)]
    draw_matrix(ax, xc, C, "C", ("cell", (0, 0)), cw=CW_C, labels=dots)

    # under the matrices: c_11 as a sum of products, each term's size
    y0 = 3 * CH + 0.95
    terms = [rf"a_{{1{k + 1}}} b_{{{k + 1}1}}" for k in range(3)]
    sizes = [F.exp_of(x[k]) + F.exp_of(y[k]) for k in range(3)]
    xs = [xb + 2.3 + k * 2.6 for k in range(3)]
    ax.text(xs[0] - 1.3, y0, "$c_{11} =$", ha="right", va="center", fontsize=16)
    for k, (t, s, tx) in enumerate(zip(terms, sizes, xs)):
        ax.text(tx, y0, f"${t}$", ha="center", va="center", fontsize=16)
        if k:
            ax.text(tx - 1.3, y0, "$+$", ha="center", va="center", fontsize=16)
        ax.text(tx, y0 + 0.65, rf"$\approx 2^{{{s}}}$", ha="center", va="center", fontsize=14,
                color="#222" if step >= 3 else MUTED)
    if step >= 3:
        ax.add_patch(Ellipse((xs[R], y0 + 0.32), 2.0, 1.55, facecolor="none", edgecolor=CONTRIB, linewidth=2.2))

    # legend, right of the sum
    lx, ly = 0.0, y0 - 0.25
    items = []
    if step >= 2:
        items.append((LARGEST, "largest in the row / column"))
    if step >= 3:
        items.append((CONTRIB, "behind the largest term of $c_{11}$"))
    for n, (color, text) in enumerate(items):
        yy = ly + n * 0.7
        ax.add_patch(Ellipse((lx, yy), 0.5, 0.36, facecolor="none", edgecolor=color, linewidth=2.2))
        ax.text(lx + 0.45, yy, text, ha="left", va="center", fontsize=13)

    x0, x1, y1, y_0 = -0.6, xc + 3 * CW_C + 0.4, y0 + 1.2, -0.95
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y_0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.set_size_inches(8.0, 8.0 * (y1 - y_0) / (x1 - x0))
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


# ---- slide 2: slices over 53 + ESC bits ----

def figure_esc_slices():
    rows = [(f"a_{{1{k + 1}}}", x[k]) for k in range(3)] + [(f"b_{{{k + 1}1}}", y[k]) for k in range(3)]
    marks = {0: LARGEST, 1: CONTRIB, 4: CONTRIB, 5: LARGEST}   # x_p, x_r, y_r, y_q
    ys = [0, 1, 2, 3.8, 4.8, 5.8]

    top = max(F.exp_of(v) for v in x + y)       # 2^0: both groups' largest leading bit
    hi = top
    lo = min(F.float_bits(v)[-1][0] for v in x + y)
    last = top - W + 1                           # the W-th bit below the leading bit
    K = W // 8
    assert 8 * K == W                            # 53 + ESC fills the slices exactly
    col = lambda p: hi - p
    y_top, y_bot = -0.15, ys[-1] + 1.15

    fig = plt.figure(figsize=(8.5, 3.9))
    left = fig.add_axes([0.0, 0.08, 0.19, 0.74])
    ax = fig.add_axes([0.19, 0.08, 0.80, 0.74])
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(y_bot + 0.1, y_top - 1.6)
    ax.axis("off")
    left.set_xlim(0, 1)
    left.set_ylim(y_bot + 0.1, y_top - 1.6)
    left.axis("off")

    # slice windows over W bits, then the dumped region
    for k in range(K):
        p0 = hi - 8 * k
        F.window(ax, k, p0, p0 - 7, hi, y_top, y_bot, label=False, cut=k > 0)
    x_end = col(last) + 1
    ax.add_patch(Rectangle((x_end, y_top), hi - lo + 1 - x_end, y_bot - y_top,
                           facecolor=DUMPED, alpha=0.08, edgecolor="none"))
    ax.plot([x_end] * 2, [y_top - 1.05, y_bot], color="#222", linewidth=1.1)
    ax.text((x_end + hi - lo + 1) / 2, y_top - 0.15, "dumped", ha="center", va="bottom",
            fontsize=11, color=DUMPED)

    # 53 + ESC, from the leading bit to the last bit kept
    F.span_arrow(ax, col(top), x_end, y_top - 0.95, rf"$53 + \mathrm{{ESC}} = {W}$ bits", fontsize=12)

    for n, ((name, v), yy) in enumerate(zip(rows, ys)):
        F.track(ax, yy, hi, lo)
        F.cells(ax, yy, [(p, b, F.slice_color((hi - p) // 8) if p >= last else DUMPED)
                         for p, b in F.float_bits(v)], hi)
        color = marks.get(n, "#222")
        left.text(0.08, yy + 0.5, f"${name}$", ha="left", va="center", fontsize=12, color=color)
        left.text(0.97, yy + 0.5, entry(v, 1), ha="right", va="center", fontsize=10.5, color=color)
    for p in range(top, lo - 1, -8):
        ax.text(col(p) + 0.5, y_bot + 0.05, f"$2^{{{p}}}$", ha="center", va="top", fontsize=8.5)
    left.text(0.08, -0.55, "row 1 of $A$", ha="left", va="center", fontsize=10, color="#555")
    left.text(0.08, ys[3] - 0.45, "column 1 of $B$", ha="left", va="center", fontsize=10, color="#555")
    for t in fig.findobj(plt.Text):   # sized for the slide: the canvas is shown at full slide width
        t.set_fontsize(t.get_fontsize() * 1.3)
    return fig


# ---- slicing: every slice uses all 8 bits ----

def figure_eight_bit_slices():
    """Row 1 of A: (a) each value's 53 bits where they sit in mantissa space,
    (b) the same bits cut into 8-bit slices from the row's largest leading
    bit down, as many slices as it takes to hold every bit."""
    top = max(F.exp_of(v) for v in x)
    lo = min(F.float_bits(v)[-1][0] for v in x)
    K = math.ceil((top - lo + 1) / 8)
    lo = top - 8 * K + 1                       # whole slices
    hi = top
    names = [f"a_{{1{k + 1}}}" for k in range(3)]

    fig = plt.figure(figsize=(8.5, 3.6))
    panels = [(0.56, "(a) Row 1 of $A$"), (0.06, "(b) Cut into 8-bit slices")]
    for bottom, title in panels:
        left = fig.add_axes([0.0, bottom, 0.17, 0.34])
        ax = fig.add_axes([0.17, bottom, 0.82, 0.34])
        for a in (left, ax):
            a.set_ylim(3.1, -1.3)
            a.axis("off")
        ax.set_xlim(0, hi - lo + 1)
        left.set_xlim(0, 1)
        sliced = title.startswith("(b)")
        if sliced:
            for k in range(K):
                F.window(ax, k, top - 8 * k, top - 8 * k - 7, hi, -0.1, 3.1, label=False, cut=k > 0)
        for j, (name, v) in enumerate(zip(names, x)):
            F.track(ax, j, hi, lo)
            F.cells(ax, j, [(p, b, F.slice_color((top - p) // 8) if sliced else F.BLUE)
                            for p, b in F.float_bits(v)], hi)
            left.text(0.08, j + 0.5, f"${name}$", ha="left", va="center", fontsize=12)
            left.text(0.97, j + 0.5, entry(v, 1), ha="right", va="center", fontsize=10.5)
        left.text(0.0, -0.95, title, ha="left", va="center", fontsize=11)
    for p in range(top, lo - 1, -8):
        ax.text(hi - p + 0.5, 3.15, f"$2^{{{p}}}$", ha="center", va="top", fontsize=8.5)
    for t in fig.findobj(plt.Text):   # sized for the slide
        t.set_fontsize(t.get_fontsize() * 1.3)
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (1, 2, 3):
        fig = figure_esc_matrix(step)
        fig.savefig(OUT / f"esc_matrix_{step}.svg")   # same canvas every step, so they stack
        print("wrote", OUT / f"esc_matrix_{step}.svg")
    fig = figure_eight_bit_slices()
    fig.savefig(OUT / "eight_bit_slices.svg", bbox_inches="tight", pad_inches=0.05)
    print("wrote", OUT / "eight_bit_slices.svg")
    fig = figure_esc_slices()
    fig.savefig(OUT / "esc_slices.svg", bbox_inches="tight", pad_inches=0.05)
    print("wrote", OUT / "esc_slices.svg")
