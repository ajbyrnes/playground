"""SVG figures for the Foundations section of the slides.

Reuses the bit-row helpers from figures.py so the slides and the paper share
one visual language: one column per power of two, higher powers on the left,
filled = 1, outlined = 0.

Run: .venv/bin/python figures/scripts/foundations.py   → writes figures/static/slides/*.svg
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figures as F
from figures import BLUE, bit_row, float_bits, power_ticks, setup, span_arrow, track

OUT = Path(__file__).parent.parent / "static" / "slides"

# Slides are viewed wider than the paper's text column: draw a bit larger.
plt.rcParams.update({"font.size": 11, "svg.fonttype": "path"})
SLIDE_W = 9.0


# ---- floating-point space: the representable numbers on a real line ----

def figure_fp_space(p=4, emin=0, emax=2):
    """A toy format with p mantissa bits, exponents emin .. emax.

    Between 2^e and 2^(e+1) there are 2^(p-1) evenly spaced values,
    spacing 2^(e-p+1); the spacing doubles at every power of two.
    """
    fig, ax = plt.subplots(figsize=(SLIDE_W, 1.9))
    shades = ["#eef4fc", "#ffffff"]
    for k, e in enumerate(range(emin, emax + 1)):
        lo, hi = 2.0**e, 2.0 ** (e + 1)
        ax.add_patch(Rectangle((lo, -0.5), hi - lo, 1.0, facecolor=shades[k % 2], edgecolor="none"))
        step = 2.0 ** (e - p + 1)
        pts = [lo + i * step for i in range(2 ** (p - 1))]
        ax.plot(pts, [0] * len(pts), "|", color=BLUE, markersize=22, markeredgewidth=2)
        ax.text((lo + hi) / 2, 0.62, f"$e = {e}$", ha="center", va="bottom")
        ax.text((lo + hi) / 2, -0.62, rf"spacing $2^{{{e - p + 1}}}$", ha="center", va="top",
                fontsize=10, color="#555")
    top = 2.0 ** (emax + 1)
    ax.plot([top], [0], "|", color=BLUE, markersize=22, markeredgewidth=2)
    ax.axhline(0, color="#999", linewidth=0.8, zorder=0)
    ax.set_xlim(2.0**emin * 0.9, top * 1.02)
    ax.set_ylim(-1.3, 1.2)
    ticks = [2.0**e for e in range(emin, emax + 2)]
    ax.set_xticks(ticks, [f"${t:g}$" for t in ticks])
    ax.set_yticks([])
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    return fig


# ---- mantissa space, built up one value at a time ----

# One running example for the three tutorial slides. x1 alone; then x2, close
# in magnitude (range 2, width 55); then x3, far below (range 55, width 108).
STEPS = [(1.751, 4), (1.317, 2), (1.092, -51)]
RANGE_COLOR = "#d9480f"


def label(ax, x0, x1, y, text, color):
    """A double-headed arrow from x0 to x1 (none if they meet), its text above
    and left-aligned with the arrow's start so short arrows don't push it out."""
    if x1 > x0:
        ax.annotate("", (x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<->", linewidth=1, color=color, shrinkA=0, shrinkB=0))
    ax.annotate(text, (x0, y), xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=13, color=color)


def figure_mantissa_step(k, steps=STEPS):
    """Mantissa space of the first k values of steps, drawn on the columns of
    the full set so the space visibly grows from slide to slide."""
    every = [m * 2.0**e for m, e in steps]
    rows = [float_bits(v) for v in every]
    hi = max(r[0][0] for r in rows)
    lo = min(r[-1][0] for r in rows)
    col = lambda q: hi - q  # left edge of power q's column

    shown = rows[:k]
    tops = [r[0][0] for r in shown]
    e_max, e_min = max(tops), min(tops)
    s_lo = min(r[-1][0] for r in shown)
    width = e_max - s_lo + 1
    erange = e_max - e_min

    n = len(steps)
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    setup(ax, hi, lo, n, [f"$x_{i + 1}$" if i < k else "" for i in range(n)])
    ax.set_ylim(n + 0.1, -2.3)
    ax.tick_params(axis="y", labelsize=14)
    for i, r in enumerate(shown):
        track(ax, i, hi, lo)
        bit_row(ax, i, r, BLUE, hi)
    power_ticks(ax, hi, lo, 16)
    ax.tick_params(axis="x", labelsize=12, length=0)

    # magnitude range: between the leading bits of the largest and smallest
    # (none for a single value: a range of 0 isn't worth a caption)
    sub = f"({e_min})" if e_min < 0 else f"{e_min}"
    if erange:
        label(ax, col(e_max), col(e_min), -0.4, rf"magnitude range $= {e_max} - {sub} = {erange}$", RANGE_COLOR)
    # width: from the top bit of the largest to the bottom bit of the smallest
    label(ax, col(e_max), col(s_lo) + 1, -1.45, rf"width $= {width}$ bits", "#222")
    # fixed margins, not a tight bbox: all three figures come out the same size
    fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.11)
    return fig


# ---- matrix product shapes: A (m x n) times B (n x p) ----

def figure_matmul_dims(m=2.0, n=3.2, p=2.0):
    """A x B = C as boxes, with the inner dimension n highlighted. Row i of A,
    column j of B and entry c_ij are shaded to tie in with the error bound."""
    N_COLOR = RANGE_COLOR
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    gap = 1.3
    ax_, bx, cx = 0.0, n + gap, n + gap + p + gap  # left edges of A, B, C
    top = n  # B is the tallest box; align all tops
    row_h, col_w = 0.45, 0.45
    shade = "#cfe0f5"

    def box(x, w, h, name):
                ax.text(x + w / 2, top + 0.25, f"${name}$", ha="center", va="bottom", fontsize=16)

    box(ax_, n, m, "A")
    box(bx, p, n, "B")
    box(cx, p, m, "C")
    # row i of A, column j of B, entry c_ij
    ri = top - 0.8
    ax.add_patch(Rectangle((ax_, ri - row_h), n, row_h, facecolor=shade, edgecolor="none", zorder=0.5))
    cj = bx + 0.9
    ax.add_patch(Rectangle((cj, 0), col_w, n, facecolor=shade, edgecolor="none", zorder=0.5))
    ax.add_patch(Rectangle((cx + 0.9, ri - row_h), col_w, row_h, facecolor=shade, edgecolor="none", zorder=0.5))
    for x0, w, h in [(ax_, n, m), (bx, p, n), (cx, p, m)]:
        ax.add_patch(Rectangle((x0, top - h), w, h, facecolor="none", edgecolor="#222", linewidth=1.4, zorder=2))
    ax.text(cx + 0.9 + col_w / 2, ri - row_h / 2, "$c_{ij}$", ha="center", va="center", fontsize=11, zorder=3)

    ax.text(ax_ + n + gap * 0.38, top - m / 2, r"$\times$", ha="center", va="center", fontsize=18)
    ax.text(bx + p + gap / 2, top - m / 2, "$=$", ha="center", va="center", fontsize=18)

    # dimension labels: m and p in grey, the inner dimension n highlighted
    def dim(x0, y0, x1, y1, text, color, off):
        ax.annotate("", (x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.2, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center", va="center",
                fontsize=14, color=color)

    grey = "#555"
    dim(ax_ - 0.25, top - m, ax_ - 0.25, top, "$m$", grey, (-0.25, 0))
    dim(ax_, top - m - 0.3, ax_ + n, top - m - 0.3, "$n$", N_COLOR, (0, -0.3))
    dim(bx - 0.25, 0, bx - 0.25, top, "$n$", N_COLOR, (-0.25, 0))
    dim(bx, -0.3, bx + p, -0.3, "$p$", grey, (0, -0.3))
    dim(cx + p + 0.25, top - m, cx + p + 0.25, top, "$m$", grey, (0.25, 0))
    dim(cx, top - m - 0.3, cx + p, top - m - 0.3, "$p$", grey, (0, -0.3))

    ax.set_xlim(-0.9, cx + p + 0.8)
    ax.set_ylim(-0.9, top + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ---- rounding in addition, at slide size ----

# ---- bit rows with significant-bit / mantissa-bit arrows (no SB/MB column) ----

SB_COLOR = "#222"
ROW_PITCH = 1.85  # one row of bits, then an arrow lane under it
LANE_SB = 1.3     # arrow height below the row's top


def bits_with_arrows(ax, y, row_bits, color, hi, lo, v):
    """A bit row, its decimal value at the right, and under it an arrow over
    the significant bits (leading 1 to last 1)."""
    F.track(ax, y, hi, lo)
    F.bit_row(ax, y, row_bits, color, hi)
    sig = F.bits(v)
    top, last = sig[0][0], sig[-1][0]
    x0, x1 = hi - top, hi - last + 1
    ax.annotate("", (x1, y + LANE_SB), xytext=(x0, y + LANE_SB),
                arrowprops=dict(arrowstyle="<->", color=SB_COLOR, linewidth=0.8, shrinkA=0, shrinkB=0,
                                mutation_scale=7))
    ax.text((x0 + x1) / 2, y + LANE_SB, f"{len(sig)} significant bits", ha="center", va="center",
            fontsize=7, color=SB_COLOR, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.annotate(F.fmt_value(v), (1, y + 0.5), xycoords=ax.get_yaxis_transform(),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center")


def rows_panel(ax, rows, labels, hi, lo, ys):
    """rows: (bits, color, value). Rows at heights ys, labels on the left."""
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(ys[-1] + ROW_PITCH - 0.2, -0.15)
    ax.set_yticks([y + 0.5 for y in ys], labels)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    for y, (rb, color, v) in zip(ys, rows):
        bits_with_arrows(ax, y, rb, color, hi, lo, v)


def decimal_header(ax):
    ax.annotate("Decimal value", (1, -0.15), xycoords=ax.get_yaxis_transform(),
                xytext=(6, 2), textcoords="offset points", ha="left", va="bottom")


def figure_addition_slide():
    """The paper's addition figure: (a) the sum fits in 53 bits, (b) it
    doesn't, so the bits past the window (red) are rounded away."""
    from fractions import Fraction
    with plt.rc_context({"font.size": 9}):
        xa, ya = F.round_bits(math.pi, 20), F.round_bits(math.e / 1024, 20)
        xb, yb = F.round_bits(math.pi, 46), F.round_bits(math.e / 1024, 46)
        pairs = [(xa, ya), (xb, yb)]
        sums = [Fraction(x) + Fraction(y) for x, y in pairs]
        lo = min(min(F.float_bits(v)[-1][0] for v in pair) for pair in pairs)
        lo = min(lo, *(F.bits(sm, width=53)[-1][0] for sm in sums))
        hi = 2
        ys = [i * ROW_PITCH for i in range(3)]
        fig, axes = plt.subplots(2, 1, figsize=(F.WIDTH_IN, 3.2), gridspec_kw={"hspace": 0.25})
        fig.subplots_adjust(left=0.1, right=0.7)
        for ax, (x, y), sm, tag in zip(axes, pairs, sums, "ab"):
            cut = F.bits(sm)[0][0] - 52
            rows = [(F.float_bits(x), F.BLUE, x), (F.float_bits(y), F.BLUE, y),
                    (F.bits(sm, width=53), lambda p, cut=cut: F.GREEN if p >= cut else F.RED, sm)]
            rows_panel(ax, rows, ["$x$", "$y$", "$x+y$"], hi, lo, ys)
            F.cut_line(ax, cut, hi, -0.1, ys[-1] + 1.1)
            ax.text(-0.075, 0.5, f"({tag})", transform=ax.transAxes, ha="right", va="center")
        F.power_ticks(axes[1], hi, lo, 8)
        decimal_header(axes[0])
        return fig


def figure_split_slide():
    """One split, x = x^(1) + x_^(2) (slice plain, remainder underlined, as in
    the matrix version), with M = 24 as in the paper."""
    with plt.rc_context({"font.size": 9}):
        x, M = math.pi, 24
        x1, x2 = (float(v) for v in F.extract_scalar(x, M))
        sigma = 2.0**M * 2.0 ** math.ceil(math.log2(abs(x)))
        cut = F.float_bits(sigma + x)[0][0] - 52
        assert x1 + x2 == x and x2 >= 0
        vals = [x, x1, x2]
        hi = F.float_bits(x)[0][0]
        lo = min(F.float_bits(v)[-1][0] for v in vals)
        ys = [i * ROW_PITCH for i in range(3)]
        fig, ax = plt.subplots(figsize=(F.WIDTH_IN, 1.8))
        fig.subplots_adjust(left=0.07, right=0.7)
        rows = [(F.float_bits(v), c, v) for v, c in zip(vals, [F.BLUE, F.GREEN, F.RED])]
        rows_panel(ax, rows, [r"$x$", r"$x^{(1)}$", r"$\underline{x}^{(2)}$"], hi, lo, ys)
        for y, op in zip(ys[1:], "=+"):
            ax.text(-2.8, y - 0.5, op, ha="center", va="center", fontsize=10)
        F.cut_line(ax, cut, hi, -0.15, ys[-1] + 1.1)
        ax.annotate("split", (hi - cut + 1, -0.15), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", annotation_clip=False)
        F.power_ticks(ax, hi, lo, 8)
        decimal_header(ax)
        return fig


# ---- the Ozaki scheme: the paper's figures, at their own 9 pt layout ----

import re


def as_running_A(make):
    """A slicing figure drawn for the vector x, relabeled for the slides'
    running example: A, a 1 x 3 matrix with x's values (a_1j, r_1j, A^(p),
    (e_A)_1). pieces_text measures each piece when it's drawn, so its pieces
    are renamed before layout; every other label is renamed afterwards."""
    PIECES = [(r"r_j = x_j \,/\, e", r"r_{1j} = a_{1j} \,/\, (e_A)_1"), ("$e = 2^", "$(e_A)_1 = 2^")]
    AFTER = [("$x_{", "$a_{1"), ("$x^{(", "$A^{("), ("(a) $x$", "(a) $A$")]

    def wrapped():
        original = F.pieces_text

        def pieces_text(ax, x, y, pieces, *args, **kw):
            renamed = []
            for text, *rest in pieces:
                for old, new in PIECES:
                    text = text.replace(old, new)
                renamed.append((text, *rest))
            return original(ax, x, y, renamed, *args, **kw)

        F.pieces_text = pieces_text
        try:
            fig = paper_figure(make)()
        finally:
            F.pieces_text = original
        for t in fig.findobj(plt.Text):
            text = t.get_text()
            for old, new in AFTER:
                text = text.replace(old, new)
            text = re.sub(r"^\$r_\{(\d)\}\$$", r"$r_{1\1}$", text)
            t.set_text(text)
        return fig
    return wrapped


def paper_figure(make):
    """The paper's figures are laid out for 9 pt text; keep that (the SVG
    scales up on the slide anyway)."""
    def wrapped():
        with plt.rc_context({"font.size": 9}):
            return make()
    return wrapped


# ---- the Ozaki scheme: splitting a 3 x 3 matrix, row by row ----

class MatrixRows:
    """Names A's entries for entries_panel: a_11, a_12, ..."""
    def entry(self, letter, g, k):
        return f"{letter}_{{{g + 1}{k + 1}}}"


def figure_slice_matrix():
    """Split_Mat on the running example, A = x as a 1 x 3 matrix: (a) each row's bits, colored by the
    slice they go to; every row has its own windows, set by its own largest
    entry; (b) the slices A^(1), A^(2), ... as FP64 matrices."""
    # wider than the paper's text column, so three 3 x 3 FP64 tiles fit on a line
    width_in, F.WIDTH_IN = F.WIDTH_IN, 8.5
    try:
        return _slice_matrix()
    finally:
        F.WIDTH_IN = width_in


def _slice_matrix():
    with plt.rc_context({"font.size": 9}):
        A = [F.slice_vector()]  # the running example: a 1 x 3 A, x's values
        m, n = len(A), len(A[0])
        b, passes, D = F.ozaki_split(A)
        hi = max(ps[0][0] for ps in passes)
        lo = min(min(F.float_bits(v)[-1][0] for v in row) for row in A)
        ys = [[g * (n + 1) + k for k in range(n)] for g in range(m)]
        height = m * (n + 1) - 1

        tile_w, tile_h = n * F.FP_CELL_W, m * F.TILE_CELL_H
        c_in = F.tiles_height_in(len(D), len(D), tile_w, tile_h, line_gap=F.SLICE_LINE_GAP)
        fig, [(left, ax)], c_ax = F.matrix_figure([height], c_in, gap_in=0.6)

        F.bits_axes(ax, height, hi, lo)
        for g, (row, ps) in enumerate(zip(A, passes)):
            y0, y1 = ys[g][0] - 0.1, ys[g][-1] + 1.1
            for k, (e, cut) in enumerate(ps):
                F.window(ax, k, e, cut, hi, y0, y1, label=False, cut=k > 0)
            for j, v in enumerate(row):
                F.cells(ax, ys[g][j], [(p, bit, F.slice_color(F.slice_of(ps, p))) for p, bit in F.float_bits(v)], hi)
            # the row's (first) splitting factor, right of the row
            E = ps[0][0]
            ax.annotate(rf"$\sigma_{g + 1} = 2^{{{b}}} \cdot 2^{{{E}}} = 2^{{{b + E}}}$",
                        (1, (y0 + y1) / 2), xycoords=ax.get_yaxis_transform(),
                        xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=10)
        F.power_ticks(ax, hi, lo, 8)
        F.entries_panel(left, MatrixRows(), A, "a", ys, height)
        left.set_title(rf"(a) $A$, split row by row ($\alpha = {53 - b}$)", loc="left", fontsize=9, pad=12)

        F.tiles(c_ax, [{"values": [[F.fp_short(v) for v in r] for r in Dk], "color": F.slice_color(k),
                        "above": f"$A^{{({k + 1})}}$"} for k, Dk in enumerate(D)],
                len(D), cell_w=F.FP_CELL_W, fontsize=F.FP_SLICE_FONT, above_size=F.SLICE_LABEL_FONT,
                line_gap=F.SLICE_LINE_GAP)
        c_ax.text(-0.8, F.CAPTION_Y, "(b) FP64 slices", ha="left", va="bottom", fontsize=9)
        return fig


# ---- slides' versions of the slicing figures ----

def figure_slice_vector_slices():
    """(c) of the Ozaki slicing figure on its own, one slice per line, for
    the right-hand column next to the mantissa-space panels: x's slices as
    FP64 vectors, every entry keeping its own exponent."""
    with plt.rc_context({"font.size": 9}):
        x = F.slice_vector()
        _, _, D = F.ozaki_split([x])
        D = [Dk[0] for Dk in D]
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        F.tiles(ax, [{"values": [[F.fp_short(v) for v in Dk]], "color": F.slice_color(k),
                      "above": f"$x^{{({k + 1})}}$"} for k, Dk in enumerate(D)],
                1, cell_w=F.FP_CELL_W, fontsize=F.FP_SLICE_FONT, above_size=F.SLICE_LABEL_FONT,
                line_gap=F.SLICE_LINE_GAP)
        ax.text(-0.8, F.CAPTION_Y, "FP64 slices", ha="left", va="bottom", fontsize=9)
        return fig


def figure_immu_blockfloat(shown=(0, 1), cell_w=0.9):
    """Generic version of the INT8 slices, in the blank-tile style of the
    product figure: an FP64 A equals a sum of INT8 slices, each a block of
    mantissa bits weighted 2^-7p, all sharing A's row exponents e_A (the
    orange strip). No values: just shapes, formats and colors."""
    with plt.rc_context({"font.size": 9}):
        blank = [[""] * 3 for _ in range(3)]
        H = 3 * F.TILE_CELL_H
        fig, ax = plt.subplots(figsize=(F.WIDTH_IN, 1.4))
        note = lambda x, y, text, color: ax.text(x, y, text, ha="center", va="top", fontsize=6.5, color=color)
        mid = 1.5 * F.TILE_CELL_H
        x = 0

        SCALE_W, GAP = 0.55, 0.22   # e_A's strip, set a little apart from the slice it scales

        def slice_tile(x, label, weight, k):
            # the slice's weight, then e_A as its own narrow strip, then the INT8 block
            ax.text(x + F.TILE_FACTOR_W - 0.15, mid, weight, ha="right", va="center", fontsize=7.5)
            x += F.TILE_FACTOR_W
            F.tile(ax, x, 0, {"values": [[""]] * 3, "color": F.E_A_COLOR, "above": ""}, SCALE_W)
            note(x + SCALE_W / 2, H + 0.15, "$e_A$", F.E_A_COLOR)
            x += SCALE_W + GAP
            F.tile(ax, x, 0, {"values": blank, "color": F.slice_color(k), "above": ""}, cell_w)
            ax.text(x + 1.5 * cell_w, -0.12, label, ha="center", va="bottom", fontsize=8.5, color=F.slice_color(k))
            note(x + 1.5 * cell_w, H + 0.15, "INT8", F.GREEN)
            return x + 3 * cell_w

        # A itself, FP64, in neutral grey so it isn't mistaken for its first (blue) slice
        F.tile(ax, x, 0, {"values": blank, "color": "#777", "above": ""}, cell_w)
        ax.text(x + 1.5 * cell_w, -0.12, "$A$", ha="center", va="bottom", fontsize=8.5, color="#444")
        note(x + 1.5 * cell_w, H + 0.15, "FP64", "#555")
        x += 3 * cell_w
        ax.text(x + 0.7, mid, "$=$", ha="center", va="center", fontsize=10)
        x += 1.4
        for n, k in enumerate(shown):
            if n:
                ax.text(x + 0.5, mid, "$+$", ha="center", va="center", fontsize=10)
                x += 1.0
            x = slice_tile(x, f"$A^{{({k + 1})}}$", f"$2^{{c_{k + 1}}}$", k)
        ax.text(x + 1.4, mid, r"$+ \;\cdots\; +$", ha="center", va="center", fontsize=10)
        x += 2.8
        x = slice_tile(x, "$A^{(s)}$", "$2^{c_s}$", len(shown))

        ax.set_xlim(-0.3, x + 0.3)
        ax.set_ylim(H + 0.8, -0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig


def figure_immu_product_generic():
    """The IMMU product figure (one slice product, INT8 x INT8 -> INT32,
    then to FP64), with generic slices A^(i), B^(j) and generic shared
    exponents 2^c_i, 2^c_j instead of the running example's 2^-7, 2^-14."""
    orig_products, orig_pieces = F.immu_products, F.pieces_text

    def products(A, B, bps=7):
        s, s2, pairs, product, operands = orig_products(A, B, bps)
        def generic(i, j):
            a, b = operands(i, j)
            a["factor"], b["factor"] = "$2^{c_i}$", "$2^{c_j}$"
            return a, b
        return s, s2, pairs, product, generic

    def pieces_text(ax, x, y, pieces, *args, **kw):   # renamed before layout, so spacing is right
        pieces = [(text.replace("2^{-14}", "2^{c_i + c_j}"), *rest) for text, *rest in pieces]
        return orig_pieces(ax, x, y, pieces, *args, **kw)

    F.immu_products, F.pieces_text = products, pieces_text
    try:
        fig = paper_figure(lambda: F.figure_product_steps("immu"))()
    finally:
        F.immu_products, F.pieces_text = orig_products, orig_pieces
    names = {"$A^{(1)}$": "$A^{(i)}$", "$B^{(1)}$": "$B^{(j)}$", "$A^{(1)} B^{(1)}$": "$A^{(i)} B^{(j)}$",
             "$A^{(1)} B^{(1)}$ in FP64": "$A^{(i)} B^{(j)}$ in FP64", "$C^{(1)}$": "$C$"}
    for t in fig.findobj(plt.Text):
        t.set_text(names.get(t.get_text(), t.get_text()))
    return fig


def narrow(make, width_in):
    """Draw a tile figure on a narrower canvas: its labels are laid out at
    draw time (pieces_text), so instead of scaling text afterwards, shrink
    the figure around fixed-size text."""
    def wrapped():
        old, F.WIDTH_IN = F.WIDTH_IN, width_in
        try:
            return make()
        finally:
            F.WIDTH_IN = old
    return wrapped


def scale_text(fig, k):
    """Every label in fig k times bigger: the paper's figures are drawn for
    9 pt text in a 6.5 in column, which comes out small next to slide text."""
    for t in fig.findobj(plt.Text):
        t.set_fontsize(t.get_fontsize() * k)
    return fig


def save_parts(fig, parts):
    """Save pieces of one figure as separate SVGs: parts maps a name to the
    indices of the axes it keeps, cropped to those axes and their labels."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for name, idx in parts.items():
        box = Bbox.union([fig.axes[i].get_tightbbox(r) for i in idx])
        box = box.transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        fig.savefig(OUT / f"{name}.svg", bbox_inches=box)
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [
        ("fp_space", figure_fp_space),
        *((f"mantissa_space_{k}", lambda k=k: figure_mantissa_step(k)) for k in (1, 2, 3)),
        ("addition_rounding", figure_addition_slide),
        ("matmul_dims", figure_matmul_dims),
        ("split_scalar", figure_split_slide),
        ("slice_vector", paper_figure(F.figure_slice_ozaki)),
        ("slice_matrix", figure_slice_matrix),
        ("slice_product", paper_figure(lambda: F.figure_product_steps("ozaki"))),
        ("span_slices", paper_figure(F.figure_span)),
        # Ozaki on IMMU (Ootomo et al.) and with ADP (Schwarz et al.)
        ("immu_slice", paper_figure(F.figure_slice_immu)),
        ("immu_product", paper_figure(lambda: F.figure_product_steps("immu"))),
        ("esc", paper_figure(F.figure_esc)),
        ("esc_slice", as_running_A(F.figure_slice_esc)),
        ("slice_vector_slices", figure_slice_vector_slices),
        ("immu_blockfloat", figure_immu_blockfloat),
    ]:
        fig = make()
        tight = not name.startswith("mantissa_space")
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight" if tight else None, pad_inches=0.05)
        print("wrote", OUT / f"{name}.svg")

    # the IMMU slicing figure again, in two pieces for two slides:
    # (a) x and (b) its rescaling; then (c) the INT8 slices
    save_parts(as_running_A(F.figure_slice_immu)(), {"immu_rescale": [0, 1, 2, 3], "immu_slices": [4]})
    # the Ozaki slicing figure's mantissa-space panels, (a) x and (b) its slice
    # windows, for the left column (the slices themselves: slice_vector_slices)
    save_parts(paper_figure(F.figure_slice_ozaki)(), {"slice_vector_bits": [0, 1, 2, 3]})

    # draft3's versions, with bigger text for the slides (files the earlier
    # drafts use are left as they were)
    for name, make, k in [
        ("addition_rounding_d3", figure_addition_slide, 1.4),
        ("split_scalar_d3", figure_split_slide, 1.4),
        ("matmul_dims_d3", figure_matmul_dims, 1.15),
        ("span_slices_d3", paper_figure(F.figure_span), 1.0),
        ("esc_d3", paper_figure(F.figure_esc), 1.15),
        ("esc_slice_d3", as_running_A(F.figure_slice_esc), 1.0),
        ("immu_product_d3", narrow(figure_immu_product_generic, 4.3), 1.0),
    ]:
        fig = scale_text(make(), k)
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
        print("wrote", OUT / f"{name}.svg")
    save_parts(scale_text(paper_figure(F.figure_slice_ozaki)(), 1.15), {"slice_vector_bits_d3": [0, 1, 2, 3]})
