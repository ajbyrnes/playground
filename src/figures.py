"""Static PDF figures for the paper, drawn in the same bit-row style as the
interactive visualizers (bits.js): one column per power of two, higher powers
on the left, filled = 1, outlined = 0, nothing = no bit there.

Run from src/:  ../.venv/bin/python figures.py   → writes figures/*.pdf
"""
import colorsys
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from splitting import beta, extract_scalar, split_mat

OUT = Path(__file__).parent / "figures"

# Same colors as bits.js. Used with one meaning in every figure:
# blue = an input, green = bits that are kept, red = bits rounded away.
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#e34948"
# the product figures' row and column exponents, e_A and e_B
E_A_COLOR, E_B_COLOR = "#d9480f", "#7048e8"
TRACK = "#f2f2f2"  # light background behind each row, to help line up columns
FADE = 0.3         # opacity of trailing 0s: in the mantissa, but not significant

# Text width of the paper (letter, 1in margins); figures are drawn at this
# size so \includegraphics[width=\textwidth] doesn't rescale the fonts.
WIDTH_IN = 6.5
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],        # Computer Modern, shipped with matplotlib
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "pdf.fonttype": 42,             # embed TrueType, so the text stays selectable
})


# ---- bits ----

def bits(v, width=None):
    """v's significant bits as [(power, bit)], leading bit first.

    v is a float or an exact Fraction (e.g. an unrounded sum). With width,
    pad with trailing 0s to that many positions, like a double's 53; without
    it, stop at the lowest 1.
    """
    v = abs(Fraction(v))
    if v == 0:
        return []
    top = math.floor(math.log2(v))
    # floor(log2) of a float can be off by one near powers of two: fix it exactly
    while Fraction(2) ** top > v:
        top -= 1
    while Fraction(2) ** (top + 1) <= v:
        top += 1
    out, p, r = [], top, v
    while r or (width and len(out) < width):
        b = int(r >= Fraction(2) ** p)
        r -= b * Fraction(2) ** p
        out.append((p, b))
        p -= 1
    return out


def float_bits(x):
    """A double's 53 significand positions, trailing 0s included."""
    return bits(x, width=53)


def n_significant(v):
    """Bits from the leading 1 to the last 1."""
    return len(bits(v))


def fmt_value(v):
    """v in decimal, for the column right of its row. A double prints as
    repr does (the shortest decimal that reads back as it); a value that
    isn't a double (an unrounded sum) gets 17 digits and ≈."""
    if Fraction(float(v)) == v:
        text, approx = repr(float(v)), ""
    else:
        with localcontext() as ctx:
            ctx.prec = 17
            text, approx = str(Decimal(v.numerator) / Decimal(v.denominator)), r"\approx "
    mantissa, _, exp = text.partition("e")
    return f"${approx}{mantissa}" + (rf"\times 10^{{{int(exp)}}}" if exp else "") + "$"


# ---- drawing ----

def bit_row(ax, y, row_bits, color, hi):
    """One row of bits at height y. Column for power p spans x = hi - p .. +1.

    color is one color for every bit, or a function of the power p. 0s after
    the last 1 are faded: the mantissa has room for them, but they aren't
    significant.
    """
    last_one = min(p for p, b in row_bits if b)
    for p, b in row_bits:
        c = color(p) if callable(color) else color
        ax.add_patch(Rectangle((hi - p + 0.08, y + 0.12), 0.84, 0.76,
                               facecolor=c if b else "white", edgecolor=c, linewidth=0.5,
                               alpha=FADE if p < last_one else 1))


# row_notes' columns: anchored at the plot's right edge, then offset by this
# many points, so the gaps are the same in every figure
NOTE_X = (5, 42)


def row_notes(ax, y, v, mantissa=53):
    """Right of the row: significant bits / mantissa width, then v in decimal."""
    notes = [f"{n_significant(v)}/{mantissa}", fmt_value(v)]
    for dx, text in zip(NOTE_X, notes):
        ax.annotate(text, (1, y + 0.5), xycoords=ax.get_yaxis_transform(),
                    xytext=(dx, 0), textcoords="offset points", ha="left", va="center")


def note_headers(ax):
    """Headers over row_notes' columns, just above the top row."""
    for dx, text in zip(NOTE_X, ["SB/MB", "Decimal value"]):
        ax.annotate(text, (1, 0), xycoords=ax.get_yaxis_transform(),
                    xytext=(dx, 2), textcoords="offset points", ha="left", va="bottom")


def track(ax, y, hi, lo):
    ax.add_patch(Rectangle((0, y + 0.06), hi - lo + 1, 0.88, facecolor=TRACK, edgecolor="none"))


def cut_line(ax, cut, hi, y0, y1):
    """Dashed line between column `cut` (kept) and column `cut - 1` (rounded away)."""
    ax.plot([hi - cut + 1] * 2, [y0, y1], color="#222", linewidth=0.9, dashes=(3, 2))


def power_ticks(ax, hi, lo, step):
    """A 2^p label under every column with p a multiple of step."""
    ps = [p for p in range(hi, lo - 1, -1) if p % step == 0]
    ax.set_xticks([hi - p + 0.5 for p in ps], [f"$2^{{{p}}}$" for p in ps])
    ax.tick_params(axis="x", length=2, width=0.5, pad=2)


def setup(ax, hi, lo, n_rows, labels):
    """Columns hi..lo, rows top to bottom, a math label left of each row."""
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(n_rows, 0)
    ax.set_yticks([r + 0.5 for r in range(n_rows)], labels)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# ---- figure 1: when addition rounds ----

def round_bits(x, n):
    """x rounded to n significant bits (the scaling by 2^k is exact)."""
    k = n - 1 - float_bits(x)[0][0]
    return round(x * 2.0**k) / 2.0**k


def addition_panel(ax, x, y, lo, show_ticks):
    """Rows x, y, and the exact x + y; the sum's bits below its 53-bit window
    (what fl would round away) in red."""
    exact = Fraction(x) + Fraction(y)
    sum_bits = bits(exact)
    top = sum_bits[0][0]
    cut = top - 52
    # in the window, show all 53 positions like a double would; below, up to the last 1
    sum_bits = bits(exact, width=53)

    hi = 2  # one column of headroom for a carry
    setup(ax, hi, lo, 3, ["$x$", "$y$", "$x+y$"])
    for r, v in enumerate([x, y, exact]):
        track(ax, r, hi, lo)
        row_notes(ax, r, v)
    bit_row(ax, 0, float_bits(x), BLUE, hi)
    bit_row(ax, 1, float_bits(y), BLUE, hi)
    bit_row(ax, 2, sum_bits, lambda p: GREEN if p >= cut else RED, hi)
    cut_line(ax, cut, hi, -0.1, 3.1)
    if show_ticks:
        power_ticks(ax, hi, lo, 8)


def figure_addition():
    # π and e/1024 rounded to fewer significant bits than a double holds.
    # (a) 20 bits: the exact sum fits in 53, so fl(x + y) = x + y.
    # (b) 46 bits: just enough that the sum needs a few more than 53 (56),
    #     without filling the operands' mantissas.
    xa, ya = round_bits(math.pi, 20), round_bits(math.e / 1024, 20)
    xb, yb = round_bits(math.pi, 46), round_bits(math.e / 1024, 46)
    assert xa + ya == Fraction(xa) + Fraction(ya)
    assert xb + yb != Fraction(xb) + Fraction(yb)

    # both panels share one column range, so the exponents line up; it reaches
    # the end of every operand's mantissa and of every exact sum
    pairs = [(xa, ya), (xb, yb)]
    lo = min(min(float_bits(v)[-1][0] for v in pair) for pair in pairs)
    lo = min(lo, *(bits(Fraction(x) + Fraction(y), width=53)[-1][0] for x, y in pairs))
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 1.75), gridspec_kw={"hspace": 0.35})
    fig.subplots_adjust(left=0.1, right=0.66)  # room on the right for row_notes
    addition_panel(axes[0], xa, ya, lo, show_ticks=False)
    addition_panel(axes[1], xb, yb, lo, show_ticks=True)
    note_headers(axes[0])
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.075, 0.5, f"({tag})", transform=ax.transAxes, ha="right", va="center")
    return fig


# ---- figure 2: one split ----

def figure_split(x=math.pi, M=24):
    """x = x_hi + x_lo for one ExtractScalar, as in extract_scalar.html."""
    x_hi, x_lo = (float(v) for v in extract_scalar(x, M))
    sigma = 2.0**M * 2.0 ** math.ceil(math.log2(abs(x)))
    cut = float_bits(sigma + x)[0][0] - 52   # lowest power fl(σ + x) can hold
    assert x_hi + x_lo == x and x_lo >= 0

    rows = [x, x_hi, x_lo]
    hi = float_bits(x)[0][0]
    lo = min(float_bits(v)[-1][0] for v in rows)
    labels = [r"$x$", r"$x_{\mathrm{hi}}$", r"$x_{\mathrm{lo}}$"]

    # a blank half row between rows for the = and + signs
    GAP = 0.75
    ys = [0, 1 + GAP, 2 + 2 * GAP]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.05))
    fig.subplots_adjust(left=0.07, right=0.66)  # room on the right for row_notes
    setup(ax, hi, lo, 3, labels)
    ax.set_ylim(ys[-1] + 1, 0)
    ax.set_yticks([y + 0.5 for y in ys], labels)

    for y, v, color in zip(ys, rows, [BLUE, GREEN, RED]):
        track(ax, y, hi, lo)
        bit_row(ax, y, float_bits(v), color, hi)
        row_notes(ax, y, v)
    for y, op in zip(ys[1:], "=+"):
        ax.text(-1.3, y - GAP / 2, op, ha="center", va="center", fontsize=10)
    cut_line(ax, cut, hi, -0.1, ys[-1] + 1.1)
    # name the cut over its top end
    ax.annotate("split", (hi - cut + 1, -0.1),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                annotation_clip=False)
    power_ticks(ax, hi, lo, 8)
    note_headers(ax)
    return fig


# ---- figure 3: exponent span vs. slice count ----

def make_x(n, span, seed=1):
    """n positive values with random 53-bit significands in [1, 2) and
    exponents spread evenly from ⌈span/2⌉ down, so max E - min E = span
    (split_vector.html's makeX, without the random signs)."""
    rng = random.Random(seed)
    return [(1 + rng.getrandbits(52) / 2**52) * 2.0 ** (math.ceil(span / 2) - round(i * span / (n - 1)))
            for i in range(n)]


def split_passes(x, b):
    """splitting.py's split_vector, keeping each pass's window for drawing:
    [(e, cut)], the window running from 2^e down to the cut (slices.js)."""
    passes, r = [], list(x)
    while any(r):
        e = math.ceil(math.log2(max(map(abs, r))))
        sigma = 2.0**b * 2.0**e
        piece = [(sigma + v) - sigma for v in r]
        r = [v - h for v, h in zip(r, piece)]
        passes.append((e, e + b - 52))
    return passes


def slice_of(passes, p):
    """Which slice a bit at power p belongs to (slices.js's sliceOf): the
    window it falls in, or for a bit in the gap above window k, slice k - 1,
    which absorbed it; trailing 0s below the last cut go to the last slice."""
    k = next((k for k, (_, cut) in enumerate(passes) if cut <= p), None)
    if k is None:
        return len(passes) - 1
    return max(k - 1, 0) if p > passes[k][0] else k


def slice_color(k):
    """slices.js's sliceColor: hues a golden angle apart, starting at blue."""
    hue = (212 + k * 137.508) % 360
    lightness = [48, 64, 36][k % 3]
    return colorsys.hls_to_rgb(hue / 360, lightness / 100, 0.7)


def bands_panel(ax, x, passes, hi, lo):
    """drawBands for one vector: a row per element, bits colored by slice
    (0s light), each slice's window shaded behind, numbered above, with the
    powers of two it spans below."""
    n = len(x)
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(n + 0.1, -0.1)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    for k, (e, cut) in enumerate(passes):
        top, low = min(e, hi), max(cut, lo)
        ax.add_patch(Rectangle((hi - top, -0.1), top - low + 1, n + 0.2,
                               facecolor=slice_color(k), alpha=0.12, edgecolor="none"))
        mid = hi - (top + low) / 2 + 0.5
        ax.text(mid, -0.25, str(k + 1), color=slice_color(k), ha="center", va="bottom", fontsize=8)
        # the powers of two its window spans
        ax.text(mid, n + 0.25, rf"$2^{{{e}}} \ldots 2^{{{cut}}}$", ha="center", va="top", fontsize=7)
        if cut - 1 >= lo:
            ax.plot([hi - cut + 1] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.6, dashes=(2, 2))

    for i, v in enumerate(x):
        for p, b in float_bits(v):
            ax.add_patch(Rectangle((hi - p, i + 0.12), 1, 0.76, facecolor=slice_color(slice_of(passes, p)),
                                   alpha=1 if b else 0.25, edgecolor="none"))


def contents_panel(ax, x, names=None):
    """The vector as m × 2^E per element, one line per row of bands_panel.
    names are math-mode labels for the elements (x_1, x_2, ... by default)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(len(x) + 0.1, -0.1)
    ax.axis("off")
    for i, v in enumerate(x):
        E = float_bits(v)[0][0]
        ax.text(0, i + 0.5, f"${names[i] if names else f'x_{i + 1}'}$", ha="left", va="center")
        # mantissa and power of two in their own columns, so each lines up
        ax.text(0.67, i + 0.5, f"${v / 2.0**E:.3f}$", ha="right", va="center")
        ax.text(0.71, i + 0.5, rf"$\times\, 2^{{{E}}}$", ha="left", va="center")


def figure_span(spans=(4, 64)):
    """A row per span: the vector on the left and its slices on the right.
    Every row's x is slice_vector()'s significands and signs, with exponents
    spread evenly from ⌈span/2⌉ down, as in make_x."""
    ms = [v / 2.0**exp_of(v) for v in slice_vector()]
    n = len(ms)
    b = beta(n)
    xs = [[m * 2.0 ** (math.ceil(span / 2) - round(i * span / (n - 1))) for i, m in enumerate(ms)] for span in spans]
    passes = [split_passes(x, b) for x in xs]
    # one column range for every row, so powers of two line up between them
    hi = max(p[0][0] for p in passes)
    lo = min(min(float_bits(v)[-1][0] for x in xs for v in x), min(p[-1][1] for p in passes))  # every last window whole

    rows = len(spans)
    fig, axes = plt.subplots(rows, 2, figsize=(WIDTH_IN, 1.3 * rows),
                             gridspec_kw={"width_ratios": [1, 5], "wspace": 0.04, "hspace": 0.95})
    for row, (x, ps, tag, span) in enumerate(zip(xs, passes, "abcdefgh", spans)):
        contents_panel(axes[row, 0], x)
        axes[row, 0].set_title(f"({tag}) magnitude range = {span}", loc="left", fontsize=9, pad=12)
        bands_panel(axes[row, 1], x, ps, hi, lo)
    assert all(e - cut + 1 == 53 - b for ps in passes for e, cut in ps)
    return fig


# ---- matrix figures: the matrices, and one layout for every scheme ----
# Slicing figures: (a) the matrix as it is, (b) the bits each slice takes,
# (c) the slices as stored. A is split by rows, B by columns.
# Product figures: (a) which slice pairs are multiplied, (b) the first
# products, (c) the error as they're summed back into C = AB.

def make_A(m, n, span):
    """m × n matrix, row i a make_x vector over span_i bits, with span_i
    growing evenly from 0 (first row) to span (last row), as split_mul.html's
    makeA does. Each row gets its own seed, so rows have different significands."""
    return [make_x(n, round(span * i / (m - 1)), seed=i + 1) for i in range(m)]


def with_tops(groups, tops):
    """groups (A's rows or B's columns) with each entry's top 7 bits below
    its group's scale e = 2^⌈log2 max|g|⌉ set to the integer in tops, so
    every scheme's first slice is a round number. The entry's own bits
    below its first slice move down past 2^-9 (relative to e) and fill the
    rest of the significand, so the bit rows stay full; the 2^-8 bit is 0,
    so truncating (immu) and rounding (ESC) give the same first slice.
    A 0 in tops leaves the entry as it is (it must already be below 2^-7 e)."""
    out = []
    for g, ks in zip(groups, tops):
        e = Fraction(2) ** math.ceil(math.log2(max(map(abs, g))))
        row = []
        for v, k in zip(g, ks):
            r = abs(Fraction(v)) / e * 128
            if k == 0:
                assert r < 1
                row.append(v)
                continue
            low = (r - math.floor(r)) / 4  # the old bits below slice 1, now below 2^-9 · 128
            row.append(math.copysign(float(e * (abs(k) + low) / 128), k))
        out.append(row)
    return out


# the first slice of A (by rows) and of B (by columns): biggest entry of each
# group 64..127, the rest as their exponents allow (0 when below 2^-7 e)
TOPS_A = [[96, 64, 80], [112, -4, 0], [64, 0, 0]]
TOPS_B = [[96, -80, 64], [0, 4, 96], [0, 0, 80]]


def signed_A(m=3, n=3, span=16):
    """The A every scheme's figures split: make_A with round first slices
    (TOPS_A) and a_22 negative, so each figure also shows how its scheme
    handles a sign."""
    return with_tops(make_A(m, n, span), TOPS_A)


def signed_B(k=3, n=3, span=16):
    """The B every scheme's figures split and multiply by A: column j is a
    make_x vector over span_j bits (0, 8, 16, like A's rows), turned upside
    down so its exponents rise down the column. Then the big entries of B's
    columns meet the small entries of A's rows, which gives A·B a nonzero
    ESC. Round first slices (TOPS_B), b_21 negative."""
    cols = [make_x(k, round(span * j / (n - 1)), seed=j + 4)[::-1] for j in range(n)]
    return transpose(with_tops(cols, TOPS_B))


def transpose(M):
    return [list(col) for col in zip(*M)]


def exp_of(v):
    return float_bits(v)[0][0]


def esc_of(A, B):
    """ESC of A·B (Guaranteed DGEMM accuracy, Section 4): the largest ESC of
    its dot products, exp(x_p) + exp(y_q) - exp(z_r) for row x of A and
    column y of B, z_i's exponent being exp(x_i) + exp(y_i)."""
    def esc(x, y):
        ex, ey = [exp_of(v) for v in x], [exp_of(v) for v in y]
        return max(ex) + max(ey) - max(a + b for a, b in zip(ex, ey))
    return max(esc(x, y) for x in A for y in transpose(B))


def exact_product(A, B):
    """A·B with every entry an exact Fraction."""
    return [[sum(Fraction(a) * Fraction(b) for a, b in zip(row, col)) for col in transpose(B)] for row in A]


# ---- drawing pieces shared by the matrix figures ----

def tiles_height_in(n_items, per_line, tile_w, tile_h, line_gap=1.0):
    """How tall tiles() comes out, in inches, at the full text width, for
    n_items tiles of tile_w × tile_h data units (see tile_size)."""
    lines = math.ceil(n_items / per_line)
    width = per_line * (tile_w + 1.3) + 0.8        # tiles()'s xlim span
    height = lines * (tile_h + line_gap) + 0.6 - (line_gap - 1.0)   # and its ylim span
    return WIDTH_IN * height / width


def matrix_figure(bits_lines, tiles_in, gap_in=0.95):
    """The panels of a slicing figure: one bits-beside-entries panel per
    entry of bits_lines (its number of lines), then tiles across the full
    width, tiles_in tall. Returns fig, a (left, bits) pair of axes per bits
    panel, and the tiles' axes."""
    rows = [0.22 * lines + 0.3 for lines in bits_lines] + [tiles_in]
    fig = plt.figure(figsize=(WIDTH_IN, sum(rows) + gap_in * (len(rows) - 1)))
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[1.2, 5], height_ratios=rows, wspace=0.04,
                            hspace=gap_in / (sum(rows) / len(rows)))   # hspace is a fraction of the mean panel height
    pairs = [(fig.add_subplot(grid[r, 0]), fig.add_subplot(grid[r, 1])) for r in range(len(bits_lines))]
    return fig, pairs, fig.add_subplot(grid[-1, :])


def bits_axes(ax, height, hi, lo):
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(height + 0.1, -0.1)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def entries_panel(ax, side, groups, letter, ys, height):
    """Left column: each entry as m × 2^E, on its line in the group."""
    ax.set_xlim(0, 1)
    ax.set_ylim(height + 0.1, -0.1)
    ax.axis("off")
    for g, group in enumerate(groups):
        for k, v in enumerate(group):
            v, y = float(v), ys[g][k] + 0.5
            E = exp_of(v)
            ax.text(0, y, f"${side.entry(letter, g, k)}$", ha="left", va="center")
            ax.text(0.72, y, f"${v / 2.0**E:.3f}$", ha="right", va="center")
            ax.text(0.76, y, rf"$\times\, 2^{{{E}}}$", ha="left", va="center")


def shifted_entries_panel(ax, side, groups, Es, letter, ys, height):
    """Like entries_panel, for r = x / e: each entry's significand as it is,
    and its exponent written as x's exponent minus its group's E (e = 2^E),
    the minus E in orange, so it's plain that dividing by e only moves the
    exponent and leaves the bits alone."""
    ax.set_xlim(0, 1)
    ax.set_ylim(height + 0.1, -0.1)
    ax.axis("off")
    for g, group in enumerate(groups):
        for k, v in enumerate(group):
            v, y = float(v), ys[g][k] + 0.5
            E = exp_of(v)
            ax.text(0, y, f"${side.entry(letter, g, k)}$", ha="left", va="center")
            ax.text(0.6, y, f"${v / 2.0**E:.3f}$", ha="right", va="center")
            sup = {"dy": 4, "fontsize": 6.5}
            pieces_text(ax, 0.63, y, [(r"$\times\, 2$", "black"), (f"${E}$", "black", sup),
                                      (f"$-{Es[g]}$", E_A_COLOR, sup)], center=False, va="center")


def window(ax, k, top, low, hi, y0, y1, number=True, label=True, cut=True, label_size=6):
    """Slice k's window, powers top..low, shaded over lines y0..y1: numbered
    above, its powers below, a dashed line on its left edge. Clipped to the
    axes' columns."""
    x_max = ax.get_xlim()[1]
    left, right = max(hi - top, 0), min(hi - low + 1, x_max)
    ax.add_patch(Rectangle((left, y0), right - left, y1 - y0,
                           facecolor=slice_color(k), alpha=0.12, edgecolor="none"))
    if number:
        ax.text((left + right) / 2, y0 - 0.15, str(k + 1), color=slice_color(k), ha="center", va="bottom", fontsize=8)
    if label:
        ax.text((left + right) / 2, y1 + 0.15, rf"$2^{{{top}}} \ldots 2^{{{low}}}$", ha="center", va="top", fontsize=label_size)
    if cut:
        ax.plot([left] * 2, [y0, y1], color="#222", linewidth=0.6, dashes=(2, 2))


def cells(ax, y, row_bits, hi):
    """One entry's bits as solid cells: row_bits is [(power, bit, color)]."""
    for p, b, color in row_bits:
        ax.add_patch(Rectangle((hi - p, y + 0.12), 1, 0.76, facecolor=color,
                               alpha=1 if b else 0.25, edgecolor="none"))


TILE_CELL_H, TILE_SCALE_W, TILE_FACTOR_W = 0.62, 1.15, 1.5


def tile_size(it, cell_w):
    """Width and height of one tile (see tile), in data units."""
    w = (TILE_FACTOR_W if it.get("factor") else 0) + (TILE_SCALE_W if it.get("scales") else 0) \
        + len(it["values"][0]) * cell_w
    h = (TILE_CELL_H if it.get("col_scales") else 0) + len(it["values"]) * TILE_CELL_H
    return w, h


def tile(ax, x0, y0, it, cell_w=1.0, fontsize=7, above_size=8.5):
    """One small matrix with its top-left corner at (x0, y0), y growing down.
    it: {values, color, above} with values the entries as text, plus
    optionally scales (a column of each row's shared exponent, grey, on the
    left) and col_scales (a row of each column's, grey, on top), each
    coloured by scale_color / col_scale_color instead if given, and factor
    (a weight written in front of the whole tile)."""
    def box(x, y, w, text, color, alpha, text_color="black"):
        ax.add_patch(Rectangle((x + 0.04, y + 0.03), w - 0.08, TILE_CELL_H - 0.06,
                               facecolor=color, alpha=alpha, edgecolor=color, linewidth=0.5))
        ax.text(x + w / 2, y + TILE_CELL_H / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)

    w, _ = tile_size(it, cell_w)
    y = y0 + (TILE_CELL_H if it.get("col_scales") else 0)
    if it.get("factor"):
        ax.text(x0 + TILE_FACTOR_W - 0.15, y + len(it["values"]) * TILE_CELL_H / 2, it["factor"],
                ha="right", va="center", fontsize=fontsize + 1)
        x0, w = x0 + TILE_FACTOR_W, w - TILE_FACTOR_W
    x = x0 + (TILE_SCALE_W if it.get("scales") else 0)
    for i, sc in enumerate(it.get("scales") or []):
        color = it.get("scale_color")
        box(x0, y + i * TILE_CELL_H, TILE_SCALE_W, sc, color or "#777", 0.12, color or "black")
    for j, sc in enumerate(it.get("col_scales") or []):
        color = it.get("col_scale_color")
        box(x + j * cell_w, y0, cell_w, sc, color or "#777", 0.12, color or "black")
    for i, row in enumerate(it["values"]):
        for j, v in enumerate(row):
            box(x + j * cell_w, y + i * TILE_CELL_H, cell_w, v, it["color"], 0.18)
    ax.text(x0 + w / 2, y0 - 0.12, it["above"], ha="center", va="bottom", fontsize=above_size, color=it["color"])


def tiles(ax, items, per_line, cell_w=1.0, fontsize=7, above_size=8.5, line_gap=1.0):
    """Small matrices in a row, + between them, per_line to a line, line_gap
    between lines (room for the labels above). items: tile's its, each
    optionally with no_plus."""
    TILE = max(tile_size(it, cell_w)[0] for it in items) + 1.3   # the widest tile plus the gap for its +
    LINE = max(tile_size(it, cell_w)[1] for it in items) + line_gap

    for slot, it in enumerate(items):
        line, pos = divmod(slot, per_line)
        x0, y0 = pos * TILE, line * LINE
        if slot and not it.get("no_plus"):
            ax.text(x0 - 0.65, y0 + tile_size(it, cell_w)[1] / 2, "+", ha="center", va="center", fontsize=10)
        tile(ax, x0, y0, it, cell_w, fontsize, above_size)

    lines = math.ceil(len(items) / per_line)
    ax.set_xlim(-0.8, per_line * TILE)
    ax.set_ylim(lines * LINE - 0.3, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")


def fp_short(v):
    """A slice entry for a tile: ±m × 2^E with 2 decimals, or 0."""
    if v == 0:
        return "$0$"
    E = exp_of(v)
    return rf"${v / 2.0**E:.2f}\!\times\!2^{{{E}}}$"


# ---- the three schemes' splits, group by group ----

def ozaki_split(groups):
    """Split_Mat on groups (rows of A or B^T): each group's passes (its slice
    windows, for drawing) and the slices D[k], group by group, as floats."""
    b = beta(len(groups[0]))
    passes = [split_passes(g, b) for g in groups]
    D = [Dk.tolist() for Dk in split_mat(groups)]
    assert all(sum(Fraction(Dk[g][k]) for Dk in D) == Fraction(v)
               for g, group in enumerate(groups) for k, v in enumerate(group))
    return b, passes, D


def immu_split(groups, bps=7, s=None):
    """SplitInt on groups: each group's exponent E_g and the integer slices
    M[p], group by group. s slices (default: as many as the longest group
    needs; one count for the whole matrix, shorter groups padded with 0s)."""
    parts = [split_int(g, bps) for g in groups]
    s = s or max(len(M) for _, M in parts)
    Es = [E for E, _ in parts]
    M = [[parts[g][1][p] if p < len(parts[g][1]) else [0] * len(groups[g]) for g in range(len(groups))]
         for p in range(s)]
    return Es, M


def esc_split(groups, W):
    """ADP's slicing on groups: each group's exponent E_g, the s8 slices
    M[k] group by group, and F, the fractional bits kept."""
    parts = [split_adp(g, W) for g in groups]
    Es = [E for E, _, _ in parts]
    F = parts[0][2]
    stored = [as_s8(M_g) for _, M_g, _ in parts]
    K = (F + 1) // 8
    M = [[stored[g][k] for g in range(len(groups))] for k in range(K)]
    for g, (_, M_g, _) in enumerate(parts):  # stored as s8, same value as the unsigned reading
        for k in range(len(groups[g])):
            assert sum(Fraction(M[p][g][k], 2 ** (7 + 8 * p)) for p in range(K)) == \
                   sum(Fraction(M_g[p][k], 2 ** (7 + 8 * p)) for p in range(K))
    return Es, M, F


# ---- slicing figures ----

class VectorSide:
    """What entries_panel needs to name entries, for one vector x: x_1, x_2, ..."""
    by_row, word = True, "vector"

    def entry(self, letter, g, k):
        return f"{letter}_{{{k + 1}}}"


def slice_vector():
    """The x every slicing figure splits: the pathological case of a matrix
    that is a vector, so one row and one e. Its entries are signed_A()'s
    second row (a mix of exponents, a negative entry, round first slices)."""
    return signed_A()[1]


def vector_panel(left, ax, x, hi, lo):
    """(a) of every slicing figure: x's entries and their bits, where they are."""
    n = len(x)
    bits_axes(ax, n, hi, lo)
    for j, v in enumerate(x):
        track(ax, j, hi, lo)
        bit_row(ax, j, float_bits(v), BLUE, hi)
    power_ticks(ax, hi, lo, 8)
    entries_panel(left, VectorSide(), [x], "x", [list(range(n))], n)
    left.set_title("(a) $x$", loc="left", fontsize=9, pad=8)


FP_CELL_W, INT_CELL_W = 3.0, 1.5   # slice tiles' cell widths, sized for their fonts
FP_SLICE_FONT, SLICE_FONT, SLICE_LABEL_FONT = 6.5, 8, 10   # FP64 entries are long: smaller, to fit 3 tiles to a line
FP_PER_LINE = 3                     # all of x's slices on one line
CAPTION_Y = -1.5                    # slice tiles' caption, above the bigger labels
SLICE_LINE_GAP = 1.5                # between lines of slice tiles: room for the bigger labels


def figure_slice_ozaki():
    """Split_Mat (Ozaki et al., Algorithm 3) on x = slice_vector():
    (a) x's slice windows, set by σ from what's left of x on every pass,
    labelled with max |x_j|, which sets the first window, (b) the slices,
    FP64 vectors whose entries keep their own exponents."""
    x = slice_vector()
    b, passes, D = ozaki_split([x])
    ps, D = passes[0], [Dk[0] for Dk in D]
    n = len(x)

    hi = ps[0][0]
    lo = min(float_bits(v)[-1][0] for v in x)
    # one line of tiles, the full height of tiles()'s ylim, so they span the whole width
    c_in = WIDTH_IN * (TILE_CELL_H + SLICE_LINE_GAP + 0.6) / (FP_PER_LINE * (3 * FP_CELL_W + 1.3) + 0.8)
    fig, [(a_left, a_ax), (left, ax)], c_ax = matrix_figure([n + 1, n + 1], c_in)
    vector_panel(a_left, a_ax, x, hi, lo)

    # (b) x cut by its windows, on the same axis as (a)
    bits_axes(ax, n, hi, lo)
    for k, (e, cut) in enumerate(ps):
        window(ax, k, e, cut, hi, -0.1, n + 0.1, cut=k > 0, label_size=8)  # wide windows: room for big labels
    for j, v in enumerate(x):
        cells(ax, j, [(p, bit, slice_color(slice_of(ps, p))) for p, bit in float_bits(v)], hi)
    left.axis("off")
    biggest = max(map(abs, x))
    E = exp_of(biggest)
    left.set_title(rf"(b) slice windows ($\alpha = {53 - b}$): the first starts at "
                   rf"$\mathrm{{max}}_j\, |x_j| = {biggest / 2.0**E:.2f}\!\times\!2^{{{E}}}$", loc="left", fontsize=9, pad=14)

    # (c) the slices: FP64, so every entry has its own exponent
    tiles(c_ax, [{"values": [[fp_short(v) for v in Dk]], "color": slice_color(k), "above": f"$x^{{({k + 1})}}$"}
                 for k, Dk in enumerate(D)], FP_PER_LINE, cell_w=FP_CELL_W, fontsize=FP_SLICE_FONT,
          above_size=SLICE_LABEL_FONT, line_gap=SLICE_LINE_GAP)
    c_ax.text(-0.8, CAPTION_Y, "(c) FP64 slices", ha="left", va="bottom", fontsize=9)
    return fig


def int_slice_tiles(ax, M, E, factor, caption, per_line=3):
    """(c) of the integer schemes: each slice a row of integers, with x's
    exponent e = 2^E in an orange strip and the slice's weight in front."""
    tiles(ax, [{"values": [[f"${v}$" for v in Mk]], "color": slice_color(k), "above": f"$x^{{({k + 1})}}$",
                "factor": f"$2^{{-{factor(k)}}}$", "scales": [f"$2^{{{E}}}$"], "scale_color": E_A_COLOR}
               for k, Mk in enumerate(M)], per_line, cell_w=INT_CELL_W, fontsize=SLICE_FONT, above_size=SLICE_LABEL_FONT,
          line_gap=SLICE_LINE_GAP)
    ax.text(-0.8, CAPTION_Y, caption, ha="left", va="bottom", fontsize=9)


INT_TILE_W = TILE_FACTOR_W + TILE_SCALE_W + 3 * INT_CELL_W   # tile_size of int_slice_tiles' tiles


def figure_slice_immu(bps=7):
    """SplitInt (Ootomo et al., Algorithm 4) on x = slice_vector(): (a) r =
    x / e, with r's exponents written as x's minus e's, cut by the fixed
    windows, (b) the slices: integers sharing e, slice p weighted 2^-7p."""
    x = slice_vector()
    Es, M = immu_split([x], bps)
    E, M = Es[0], [Mk[0] for Mk in M]
    s, n = len(M), len(x)
    r = [Fraction(v) / Fraction(2) ** E for v in x]

    fig, [(a_left, a_ax), (left, ax)], c_ax = matrix_figure([n + 1, n + 1],
                                                           tiles_height_in(s, 3, INT_TILE_W, TILE_CELL_H, SLICE_LINE_GAP))
    vector_panel(a_left, a_ax, x, exp_of(max(map(abs, x))), min(float_bits(v)[-1][0] for v in x))

    # (b) r below the binary point, cut by the fixed windows
    hi, lo = 0, -s * bps
    bits_axes(ax, n, hi, lo)
    for k in range(s):
        window(ax, k, -k * bps - 1, -(k + 1) * bps, hi, -0.1, n + 0.1, cut=k > 0)
    ax.plot([hi + 1] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.9)  # the binary point
    for j in range(n):
        cells(ax, j, [(p, bit, slice_color((-p - 1) // bps)) for p, bit in float_bits(r[j])], hi)
    shifted_entries_panel(left, VectorSide(), [x], [E], "r", [list(range(n))], n)
    pieces_text(left, 0, -0.75, [(r"(b) $r_j = x_j \,/\, e$, with ", "black"), (f"$e = 2^{{{E}}}$", E_A_COLOR),
                                 (r", so the largest leading bit lands at $2^{-1}$", "black")],
                center=False, va="baseline", fontsize=9)

    int_slice_tiles(c_ax, M, E, lambda k: (k + 1) * bps, f"(c) INT8 slices, slice $p$ weighted $2^{{-{bps}p}}$")
    return fig


def figure_slice_esc(esc=0):
    """ADP's slicing (Guaranteed DGEMM accuracy) on x = slice_vector(), with
    W = 53 + ESC: (a) x, (b) r = x / e in two's complement, cut into 8-bit
    slices with the sign only in the first, bits past the last slice
    dropped, (c) the slices, all stored as s8, sharing e, outlined where
    that storage changed the value the bits in (b) read as (a chunk of
    128..255 stored as itself - 256, with +1 carried into the slice above). ESC belongs to the
    product x is in, not to x; it's whatever the example needs, here 0, so
    ADP takes fewer slices than SplitInt needs to keep all of x."""
    x = slice_vector()
    W = 53 + esc
    Es, M, F = esc_split([x], W)
    E, M = Es[0], [Mk[0] for Mk in M]
    K, n = len(M), len(x)
    r = [Fraction(v) / Fraction(2) ** E for v in x]

    fig, [(a_left, a_ax), (left, ax)], c_ax = matrix_figure([n + 1, n + 1],
                                                           tiles_height_in(K, 3, INT_TILE_W, TILE_CELL_H, SLICE_LINE_GAP))
    vector_panel(a_left, a_ax, x, exp_of(max(map(abs, x))), min(float_bits(v)[-1][0] for v in x))

    # (b) two's complement fixed point: sign at 2^0, slices of 8, dropped bits dark
    bits = [twos_complement_bits(v, -F) for v in r]
    hi, lo = 0, min(p for rb in bits for p, _ in rb)
    bits_axes(ax, n, hi, lo)
    for k in range(K):
        window(ax, k, -8 * k, -8 * k - 7, hi, -0.1, n + 0.1, cut=k > 0)
    ax.plot([hi + 1] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.9)
    x_W = hi + W + 1  # right edge of 2^-W: W bits from the largest leading bit, 2^-1
    ax.plot([x_W] * 2, [-0.1, n + 0.1], color="#888", linewidth=0.8, dashes=(1, 2))
    for j in range(n):
        cells(ax, j, [(p, bit, slice_color(-p // 8) if p >= -F else "#222") for p, bit in bits[j]], hi)
    # outline each slice of each entry whose s8 storage isn't what its bits read: a chunk of 128..255
    # stored as itself - 256, or a +1 carried up from the slice below
    raw = split_adp(x, W)[1]   # each slice as its bits read: slice 1 signed, the rest 0..255
    for k in range(K):
        for j in range(n):
            if raw[k][j] != M[k][j]:
                ax.add_patch(Rectangle((hi + 8 * k + 0.2, j + 0.08), 7.6, 0.84, facecolor="none",  # inset: neighbours stay apart
                                       edgecolor="#555", linewidth=0.7))
    shifted_entries_panel(left, VectorSide(), [x], [E], "r", [list(range(n))], n)
    pieces_text(left, 0, -1.35, [(r"(b) $r_j = x_j \,/\, e$, with ", "black"), (f"$e = 2^{{{E}}}$", E_A_COLOR),
                                 (r", so the largest leading bit lands at $2^{-1}$", "black")],
                center=False, va="baseline", fontsize=9)
    left.text(0, -0.75, rf"$\mathrm{{ESC}} = {esc}$; boxed: slices re-encoded in two's complement when stored "
                        "as signed int8", ha="left", va="baseline", fontsize=8, color="#555")

    int_slice_tiles(c_ax, M, E, lambda k: 8 * k + 7,
                    "(c) INT8 slices, slice $k$ weighted $2^{-(8k-1)}$")
    return fig


# ---- product figures ----

def ozaki_products(A, B):
    """EFT_Mul (Ozaki et al., Algorithm 4): every slice of A times every
    slice of B, each product exact in FP64. Returns the slice counts, the
    pairs in the order they're summed, and each pair's product (its value,
    as floats) and tile."""
    _, _, D = ozaki_split(A)
    _, _, Et = ozaki_split(transpose(B))
    E = [transpose(Ek) for Ek in Et]
    pairs = sorted(((i, j) for i in range(len(D)) for j in range(len(E))), key=lambda p: (sum(p), p[0]))
    def product(i, j):
        P = (np.array(D[i]) @ np.array(E[j])).tolist()
        assert P == [[float(v) for v in row] for row in exact_product(D[i], E[j])]  # no rounding
        return P, {"values": [[fp_short(v) for v in row] for row in P]}
    def operands(i, j):
        return ({"values": [[fp_short(v) for v in row] for row in D[i]]},
                {"values": [[fp_short(v) for v in row] for row in E[j]]})
    return len(D), len(E), pairs, product, operands


def immu_products(A, B, bps=7):
    """Algorithm 3 (Ootomo et al.): the pairs with i + j ≤ s + 1, each an
    exact integer product A^(i) B^(j), scaled by 2^-(i+j)α e_A e_B^T."""
    EA, MA = immu_split(A, bps)
    EB, MBt = immu_split(transpose(B), bps)
    s = max(len(MA), len(MBt))
    EA, MA = immu_split(A, bps, s)
    EB, MBt = immu_split(transpose(B), bps, s)
    MB = [transpose(M) for M in MBt]
    pairs = [(i, j) for t in range(s) for i in range(t + 1) for j in [t - i]]  # i + j ≤ s - 1, 0-indexed
    def product(i, j):
        P = [[sum(a * b for a, b in zip(row, col)) for col in transpose(MB[j])] for row in MA[i]]
        value = [[P[r][c] * 2.0 ** (EA[r] + EB[c] - (i + j + 2) * bps) for c in range(len(P[0]))] for r in range(len(P))]
        return value, {"values": [[f"${v}$" for v in row] for row in P],
                       "scales": [f"$2^{{{E}}}$" for E in EA], "col_scales": [f"$2^{{{E}}}$" for E in EB],
                       "weight": (i + j + 2) * bps}
    def operands(i, j):
        return ({"values": [[f"${v}$" for v in row] for row in MA[i]], "factor": f"$2^{{-{(i + 1) * bps}}}$",
                 "scales": [f"$2^{{{E}}}$" for E in EA], "scale_color": E_A_COLOR},
                {"values": [[f"${v}$" for v in row] for row in MB[j]], "factor": f"$2^{{-{(j + 1) * bps}}}$",
                 "col_scales": [f"$2^{{{E}}}$" for E in EB], "col_scale_color": E_B_COLOR})
    return s, s, pairs, product, operands


def esc_products(A, B):
    """ADP's slices (W = 53 + ESC from A·B), multiplied pairwise like
    Algorithm 3 of Ootomo et al. (i + j ≤ K + 1): each an exact integer
    product, scaled by its rows' and columns' shared exponents."""
    W = 53 + esc_of(A, B)
    EA, MA, F = esc_split(A, W)
    EB, MBt, _ = esc_split(transpose(B), W)
    MB = [transpose(M) for M in MBt]
    K = len(MA)
    pairs = [(i, j) for t in range(K) for i in range(t + 1) for j in [t - i]]
    weight = lambda k: 8 * k + 7  # slice k's integers count units of 2^-(8k+7)
    def product(i, j):
        P = [[sum(a * b for a, b in zip(row, col)) for col in transpose(MB[j])] for row in MA[i]]
        value = [[P[r][c] * 2.0 ** (EA[r] + EB[c] - weight(i) - weight(j)) for c in range(len(P[0]))]
                 for r in range(len(P))]
        return value, {"values": [[f"${v}$" for v in row] for row in P],
                       "scales": [f"$2^{{{E}}}$" for E in EA], "col_scales": [f"$2^{{{E}}}$" for E in EB],
                       "weight": weight(i) + weight(j)}
    def operands(i, j):
        return ({"values": [[f"${v}$" for v in row] for row in MA[i]], "factor": f"$2^{{-{weight(i)}}}$",
                 "scales": [f"$2^{{{E}}}$" for E in EA], "scale_color": E_A_COLOR},
                {"values": [[f"${v}$" for v in row] for row in MB[j]], "factor": f"$2^{{-{weight(j)}}}$",
                 "col_scales": [f"$2^{{{E}}}$" for E in EB], "col_scale_color": E_B_COLOR})
    return K, K, pairs, product, operands


def figure_product(scheme):
    """C = AB from the slices: (a) the grid of slice pairs A^(i) B^(j), the
    multiplied ones shaded by their product's size, (b) the first products,
    (c) the error of C as the products are summed in FP64, against the exact
    AB, with native FP64 GEMM for reference."""
    A, B = signed_A(), signed_B()
    KA, KB, pairs, product, _ = {"ozaki": ozaki_products, "immu": immu_products, "esc": esc_products}[scheme](A, B)
    exact = exact_product(A, B)
    m, n = len(A), len(B[0])

    # sum the products in FP64, in order, tracking the worst relative error
    C = [[0.0] * n for _ in range(m)]
    errors, sizes = [], {}
    rel_err = lambda C: max(abs(Fraction(C[r][c]) - exact[r][c]) / abs(exact[r][c]) for r in range(m) for c in range(n))
    for i, j in pairs:
        value, _ = product(i, j)
        sizes[i, j] = max((exp_of(v) for row in value for v in row if v), default=None)
        C = [[C[r][c] + value[r][c] for c in range(n)] for r in range(m)]
        errors.append(rel_err(C))
    gemm = [[sum(a * b for a, b in zip(row, col)) for col in transpose(B)] for row in A]  # native FP64, in order
    gemm_err = rel_err(gemm)

    fig = plt.figure(figsize=(WIDTH_IN, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1.25, 1], wspace=0.35, hspace=0.3)

    # (a) the slice pairs: multiplied (shaded darker for bigger products) or skipped
    ax = fig.add_subplot(grid[0, 0])
    done = [s for s in sizes.values() if s is not None]
    top, bottom = max(done), min(done)
    for i in range(KA):
        for j in range(KB):
            if (i, j) in sizes:
                size = sizes[i, j]
                shade = 0.15 + 0.75 * (size - bottom) / max(top - bottom, 1) if size is not None else 0
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=BLUE, alpha=shade, edgecolor="white", linewidth=1))
                ax.text(j + 0.5, i + 0.5, "0" if size is None else f"${size}$", ha="center", va="center",
                        fontsize=5.5 if KA > 6 else 7, color="white" if shade > 0.55 else "#222")
            else:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor="#bbb", hatch="////",
                                       linewidth=0.5))
    ax.set_xlim(0, KB)
    ax.set_ylim(KA, 0)
    ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(KB)], [f"${j + 1}$" for j in range(KB)], fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(KA)], [f"${i + 1}$" for i in range(KA)], fontsize=7)
    ax.tick_params(length=0, pad=2, labeltop=True, labelbottom=False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel(r"slice $j$ of $B$", fontsize=8, labelpad=3)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"slice $i$ of $A$", fontsize=8, labelpad=3)
    ax.set_title(r"(a) $A^{(i)} B^{(j)}$", loc="left", fontsize=9, pad=24)

    # (c) the error of C after each product is added
    ax = fig.add_subplot(grid[0, 1])
    floor = 2.0 ** -70  # an exact C has no error; draw it on the axis floor
    xs = range(1, len(errors) + 1)
    ax.step(xs, [max(float(e), floor) for e in errors], where="post", color=BLUE, linewidth=1.3)
    ax.axhline(float(gemm_err), color="#888", linewidth=0.9, dashes=(4, 3))
    ax.text(len(errors), float(gemm_err), "native FP64 GEMM", ha="right", va="bottom", fontsize=7, color="#666")
    ax.axhline(2.0 ** -53, color="#bbb", linewidth=0.8, dashes=(1, 2))
    ax.text(1, 2.0 ** -53, "$u = 2^{-53}$", ha="left", va="top", fontsize=7, color="#888")
    ax.set_yscale("log", base=2)
    ax.set_ylim(floor / 2, 2.0)
    ax.set_xlim(0.5, len(errors) + 0.5)
    ax.tick_params(labelsize=7, length=2, width=0.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_xlabel("products summed", fontsize=8)
    ax.set_ylabel("max relative error of $C$", fontsize=8)
    ax.set_title("(c)", loc="left", fontsize=9)

    # (b) the first products: the pairs with i + j ≤ 3 (1-indexed)
    ax = fig.add_subplot(grid[1, :])
    first = [p for p in pairs if sum(p) <= 1]
    items = []
    for i, j in first:
        _, tile = product(i, j)
        items.append({**tile, "color": "#555", "above": f"$A^{{({i + 1})}} B^{{({j + 1})}}$", "no_plus": True})
    fp = scheme == "ozaki"
    tiles(ax, items, len(items), cell_w=2.4 if fp else 1.3, fontsize=6 if fp else 6.5)
    ax.text(-0.8, -0.9, "(b) " + ("FP64 products, exact" if fp else
                                  "INT32 products: row exponents of $A^{(i)}$, column exponents of $B^{(j)}$"),
            ha="left", va="bottom", fontsize=9)
    return fig


def pieces_text(ax, x, y, pieces, center=True, **kw):
    """Text in several colours on one line at (x, y), centred there or
    (center=False) starting there: pieces is [(text, color)] or
    [(text, color, {"dy": points up, "fontsize": ...})], laid out left to
    right. Matplotlib can't colour part of one string."""
    renderer = ax.figure.canvas.get_renderer()
    texts, width = [], 0
    for text, color, *opts in pieces:
        opts = opts[0] if opts else {}
        t = ax.text(x, y, text, ha="left", color=color, **{**kw, **{k: v for k, v in opts.items() if k != "dy"}})
        t.set_transform(offset_copy(ax.transData, fig=ax.figure, x=width, y=opts.get("dy", 0), units="points"))
        width += t.get_window_extent(renderer).width * 72 / ax.figure.dpi  # points, so savefig's dpi doesn't matter
        texts.append(t)
    if center:
        for t in texts:  # shift the whole line left by half its width
            t.set_transform(offset_copy(t.get_transform(), fig=ax.figure, x=-width / 2, units="points"))


def figure_product_steps(scheme, shown=((0, 0),)):
    """Products A^(i) B^(j), one per line (default just A^(1) B^(1)), followed from the slices to C: the
    slices as stored, their product in the GEMM's own output format, and
    (for the integer schemes) that product converted to FP64 and scaled by
    its rows' and columns' exponents. Everything up to the FP64 tile is
    exact, so each product is stored as is: the k-th pair summed is C^(k),
    and only adding up C = C^(1) + C^(2) + ... rounds. Abstract: the tiles
    are blank, just their shapes, formats and the flow between them."""
    A, B = signed_A(), signed_B()
    _, _, pairs, product, operands = {"ozaki": ozaki_products, "immu": immu_products, "esc": esc_products}[scheme](A, B)
    integer = scheme != "ozaki"
    fp_w = int_w = out_w = 0.9           # cell widths: no text in them, so all square-ish
    GAP = 1.4                            # room for an operator between tiles

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.45 * len(shown)))  # bbox_inches="tight" trims the height
    note = lambda x, y, text, color: ax.text(x, y, text, ha="center", va="top", fontsize=6.5, color=color)
    exact_note = lambda x, y, fmt: note(x, y, f"{fmt}, exact", GREEN)
    head = TILE_CELL_H if integer else 0   # room for B's col_scales above the entries
    LINE = 4.0 + head
    for line, (i, j) in enumerate(shown):
        y0 = line * LINE
        value, out = product(i, j)
        Ai, Bj = operands(i, j)
        Ai.update(color=slice_color(i), above=f"${'A'}^{{({i + 1})}}$")
        Bj.update(color=slice_color(j), above=f"${'B'}^{{({j + 1})}}$")
        out.update(color="#555", above=f"$A^{{({i + 1})}} B^{{({j + 1})}}$")
        in_w = int_w if integer else fp_w
        steps = [(Ai, in_w, "INT8" if integer else "FP64"), "×", (Bj, in_w, "INT8" if integer else "FP64"), "="]
        if integer:
            # the integer product, then that product as FP64: each entry times its row's and column's exponent
            assert all(abs(v) < 2 ** 53 for row in out["values"] for v in [int(row[0].strip("$"))])
            fp = {"values": [[fp_short(v) for v in row] for row in value], "color": "#555",
                  "above": f"$A^{{({i + 1})}} B^{{({j + 1})}}$ in FP64"}
            out.update(scale_color=E_A_COLOR, col_scale_color=E_B_COLOR)
            steps += [(out, out_w, "INT32"), "→", (fp, fp_w, "FP64")]
        else:
            out["values"] = [[fp_short(v) for v in row] for row in value]
            steps += [(out, fp_w, "FP64")]
        steps += ["C"]
        blank = lambda it: {**it, **{key: [[""] * len(row) for row in it[key]] if key == "values" else [""] * len(it[key])
                                     for key in ("values", "scales", "col_scales") if it.get(key)}}
        steps = [(blank(st[0]), *st[1:]) if isinstance(st, tuple) else st for st in steps]

        x = 0
        mid = y0 + head + 1.5 * TILE_CELL_H   # the middle of the entries, below any col_scales
        for step in steps:
            if step == "C":
                ax.annotate("", xy=(x + 1.6, mid), xytext=(x + 0.15, mid),
                            arrowprops=dict(arrowstyle="-|>", color="#444", linewidth=0.8))
                ax.text(x + 1.75, mid, f"$C^{{({pairs.index((i, j)) + 1})}}$", ha="left", va="center", fontsize=9)
                x += 3.2
            elif step == "→":
                # the grey exponents are e_A and e_B^T; times the slices' weights, entrywise (∘), they're the whole scale
                ARROW = 4.2
                ax.annotate("", xy=(x + ARROW - 0.15, mid), xytext=(x + 0.4, mid),
                            arrowprops=dict(arrowstyle="-|>", color="#444", linewidth=0.8))
                pieces_text(ax, x + ARROW / 2 + 0.12, mid - 0.25,
                            [(rf"$\circ\, 2^{{-{out['weight']}}}\,$", "#222"), ("$e_A$", E_A_COLOR),
                             ("$e_B^{T}$", E_B_COLOR)], va="bottom", fontsize=7)
                x += ARROW
            elif isinstance(step, str):
                ax.text(x + GAP / 2, mid, f"${step}$", ha="center", va="center", fontsize=10)
                x += GAP
            else:
                it, cw, fmt = step
                w, th = tile_size(it, cw)
                top = y0 + (0 if it.get("col_scales") else head)   # line every tile's entries up
                tile(ax, x, top, {**it, "above": ""}, cw, fontsize=6 if cw == fp_w else 6.5)
                body = x + (TILE_FACTOR_W if it.get("factor") else 0)   # where the tile starts, after its factor
                ax.text((body + x + w) / 2, y0 - 0.12, it["above"], ha="center", va="bottom", fontsize=8.5, color=it["color"])
                entries = body + (TILE_SCALE_W if it.get("scales") else 0) + len(it["values"][0]) * cw / 2
                exact_note(entries, top + th + 0.15, fmt)
                x += w
        width = x

    ax.set_xlim(-0.3, width)
    ax.set_ylim((len(shown) - 1) * LINE + head + 3 * TILE_CELL_H + 0.8, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ---- SplitInt (Ootomo et al., Algorithm 4) on one vector ----

def split_int(x, bps):
    """SplitInt on one row x: downscale by e = 2^⌈log2 max|x|⌉, then cut
    R = x / e into fixed bps-bit windows below the binary point, by
    truncation. Returns (E, M) with e = 2^E and M[p][i] the signed integer
    slice p + 1 of x_i, enough slices to keep every bit, so that
    x_i = e · Σ_p 2^(-(p+1)·bps) M[p][i] exactly."""
    E = math.ceil(math.log2(max(map(abs, x))))
    R = [Fraction(v) / Fraction(2) ** E for v in x]  # exact: only the exponent changes
    low = min(p for r in R for p, b in bits(r) if b)  # lowest 1 in any R_i
    s = math.ceil(-low / bps)
    M = []
    for p in range(1, s + 1):
        # the bits of |R_i| from 2^-((p-1)·bps + 1) down to 2^-(p·bps), as an integer
        M.append([int(math.copysign(math.floor(abs(r) * 2 ** (p * bps)) % 2**bps, r)) for r in R])
    for i, v in enumerate(x):
        assert Fraction(2) ** E * sum(Fraction(M[p][i], 2 ** ((p + 1) * bps)) for p in range(s)) == v
    return E, M


def figure_split_int(n=4, span=10, bps=7):
    """(a) x, (b) R = x / e with SplitInt's fixed windows, (c) the slices as
    block-float: one shared exponent and a column of integers each."""
    x = make_x(n, span)
    x[2] = -x[2]  # one negative entry: slices keep x's sign
    E, M = split_int(x, bps)
    s = len(M)

    # columns are R's powers of two, 2^0 (empty: just above the binary point)
    # down to the last slice's lowest bit; x's power q sits in R's column q - E
    hi, lo = 0, -s * bps
    col = lambda p_R: hi - p_R  # left edge of R's power p_R
    slice_of_R = lambda p_R: (-p_R - 1) // bps

    fig = plt.figure(figsize=(WIDTH_IN, 4.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 5], wspace=0.04, hspace=0.9)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(3)]

    def bits_axes(ax):
        ax.set_xlim(0, hi - lo + 1)
        ax.set_ylim(n + 0.1, -0.1)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)

    # (a) x where it is, e marked just above the largest leading bit
    ax = axes[0][1]
    bits_axes(ax)
    for i, v in enumerate(x):
        track(ax, i, hi, lo)
        bit_row(ax, i, float_bits(v), BLUE, hi + E)  # hi + E: x's power q lands in column q - E
    ax.plot([col(0) + 1] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.9)
    ax.text(col(0) + 1, -0.25, f"$e = 2^{{{E}}}$", ha="center", va="bottom")
    ps = [q for q in range(hi + E, lo + E - 1, -1) if q % 8 == 0]
    ax.set_xticks([col(q - E) + 0.5 for q in ps], [f"$2^{{{q}}}$" for q in ps])
    ax.tick_params(axis="x", length=2, width=0.5, pad=2)
    contents_panel(axes[0][0], x)
    axes[0][0].set_title("(a) $x$", loc="left", fontsize=9, pad=12)

    # (b) R = x / e: the same bits, now below the binary point, cut into fixed windows
    ax = axes[1][1]
    bits_axes(ax)
    for k in range(s):
        top, low = -k * bps - 1, -(k + 1) * bps
        mid = (col(top) + col(low) + 1) / 2
        ax.add_patch(Rectangle((col(top), -0.1), bps, n + 0.2,
                               facecolor=slice_color(k), alpha=0.12, edgecolor="none"))
        ax.text(mid, -0.25, str(k + 1), color=slice_color(k), ha="center", va="bottom", fontsize=8)
        ax.text(mid, n + 0.25, rf"$2^{{{top}}} \ldots 2^{{{low}}}$", ha="center", va="top", fontsize=6.5)
        if k:
            ax.plot([col(top)] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.6, dashes=(2, 2))
    ax.plot([col(0) + 1] * 2, [-0.1, n + 0.1], color="#222", linewidth=0.9)  # the binary point
    for i, v in enumerate(x):
        R = Fraction(v) / Fraction(2) ** E
        for p, b in float_bits(R):
            ax.add_patch(Rectangle((col(p), i + 0.12), 1, 0.76, facecolor=slice_color(slice_of_R(p)),
                                   alpha=1 if b else 0.25, edgecolor="none"))
    contents_panel(axes[1][0], [v / 2.0**E for v in x], names=[f"R_{i + 1}" for i in range(n)])
    axes[1][0].set_title(rf"(b) $R = x\,/\,e$", loc="left", fontsize=9, pad=12)

    # (c) block-float: under each window, its integers and their shared exponent
    ax = axes[2][1]
    bits_axes(ax)
    ax.set_ylim(n + 0.9, -0.1)
    for k, Mk in enumerate(M):
        x0 = col(-k * bps - 1) + 0.4
        for i, m in enumerate(Mk):
            ax.add_patch(Rectangle((x0, i + 0.08), bps - 0.8, 0.84, facecolor=slice_color(k), alpha=0.18,
                                   edgecolor=slice_color(k), linewidth=0.6))
            ax.text(x0 + (bps - 0.8) / 2, i + 0.52, f"${m}$", ha="center", va="center", fontsize=8)
        ax.text(x0 + (bps - 0.8) / 2, n + 0.2, rf"$\times 2^{{{E - (k + 1) * bps}}}$",
                ha="center", va="top", fontsize=8)
    axes[2][0].axis("off")
    axes[2][0].set_ylim(n + 0.9, -0.1)
    for i in range(n):
        axes[2][0].text(0, i + 0.5, f"$x_{i + 1}$", ha="left", va="center")
    axes[2][0].set_title(f"(c) INT8 slices", loc="left", fontsize=9, pad=12)
    return fig


# ---- ADP's slicing (Guaranteed DGEMM accuracy, Section 3) on one row ----

def split_adp(x, W):
    """ADP's slicing of one row: scale by e = 2^E so the row's largest leading
    bit lands at 2^-1, then write R = x / e in two's complement with
    F = 8K - 1 fractional bits, K = ⌈(W + 1) / 8⌉, rounding toward -∞. Slice 1
    is the sign and the top 7 bits; every later slice is the next 8 bits, read
    as unsigned (0..255): the sign is stored once. Bits below 2^-F are dropped.
    Returns (E, M, F) with M[k] slice k + 1's integers, in that unsigned reading."""
    K = math.ceil((W + 1) / 8)
    F = 8 * K - 1
    E = max(float_bits(v)[0][0] for v in x) + 1
    N = [math.floor(Fraction(v) / Fraction(2) ** E * 2**F) for v in x]  # all F bits, as one integer
    M = [[n >> (F - 7) for n in N]] + [[(n >> (F - 7 - 8 * k)) & 255 for n in N] for k in range(1, K)]
    for v, n, *ms in zip(x, N, *M):
        assert sum(Fraction(mk, 2 ** (7 + 8 * k)) for k, mk in enumerate(ms)) == Fraction(n, 2**F)
        assert 0 <= Fraction(v) / Fraction(2) ** E - Fraction(n, 2**F) < Fraction(1, 2**F)
    return E, M, F


def as_s8(M):
    """ADP's production storage of split_adp's slices: every slice signed 8-bit.
    An unsigned slice x in 128..255 is stored as x - 256, which has the same
    8-bit pattern, and the slice above gets +1 to make up the 256. Works from
    the last slice up, so a carry that lifts a slice to 128 is handled in turn."""
    M = [list(row) for row in M]
    for k in range(len(M) - 1, 0, -1):
        for i, v in enumerate(M[k]):
            if v >= 128:
                M[k][i] -= 256
                M[k - 1][i] += 1
    assert all(-128 <= v <= 127 for row in M for v in row)
    return M


def twos_complement_bits(r, lo):
    """r in [-1, 1) as two's complement: the sign bit (weight -2^0) and the
    bits of r + sign at 2^-1 .. 2^lo, then on past lo to r's last 1.
    Returns [(power, bit)] from 2^0 down."""
    sign = int(r < 0)
    frac = r + sign  # in [0, 1): the bits after the sign
    last = min(lo, bits(frac)[-1][0]) if frac else lo
    return [(0, sign)] + [(p, int(frac * 2**-p) % 2) for p in range(-1, last - 1, -1)]


# ---- figure 7: ESC (Guaranteed DGEMM accuracy, Section 4) ----

def span_arrow(ax, x0, x1, y, text, fontsize=8, color="#222"):
    """A double-headed arrow from x0 to x1 at height y, its text just above
    (2 points up, whichever way the y axis runs)."""
    ax.annotate("", (x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", linewidth=0.8, color=color, shrinkA=0, shrinkB=0))
    ax.annotate(text, ((x0 + x1) / 2, y), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize, color=color)


def figure_esc(esc=12, seed=1):
    """ESC for one dot product x · y (Guaranteed DGEMM accuracy, Section 4),
    with esc.html's vectors. (a) x and (b) y: each entry's bits, blue within
    W = 53 + ESC of its vector's largest leading bit, grey beyond; in the
    target pairs r the small entry sits ESC bits down, so its 53 bits end
    exactly at W. (c) the products' exponents exp(x_i) + exp(y_i) on the same
    columns: ESC is the gap from exp(x_p) + exp(y_q) to the largest, exp(z_r)."""
    # esc.html's makeVectors: pairs 1 and 2 anti-aligned, so x's biggest entry
    # meets a small y and y's biggest a small x; pairs 3 and 4 further down
    rng = random.Random(seed)
    ex, ey = [0, -esc, -esc - 4, -1], [-esc, 0, -2, -esc - 10]
    x = [(1 + rng.getrandbits(52) / 2**52) * 2.0**e for e in ex]
    y = [(1 + rng.getrandbits(52) / 2**52) * 2.0**e for e in ey]
    ez = [a + b for a, b in zip(ex, ey)]
    xp, yq, zr = max(ex), max(ey), max(ez)
    rs = [i for i, e in enumerate(ez) if e == zr]  # r isn't unique
    assert xp + yq - zr == esc
    W = 53 + esc
    n = len(x)

    # one set of columns for all three panels: bits, and products' exponents
    hi = max(xp, yq)
    lo = min(min(float_bits(v)[-1][0] for v in x + y), min(ez))
    col = lambda p: hi - p  # left edge of power p's column
    GOLD = "#d49a00"        # the target pairs r

    fig = plt.figure(figsize=(WIDTH_IN, 4.1))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 5], height_ratios=[1, 1, 0.5], wspace=0.04, hspace=0.45)

    for row, (vals, top, name) in enumerate([(x, xp, "x"), (y, yq, "y")]):
        ax = fig.add_subplot(grid[row, 1])
        bits_axes(ax, n, hi, lo)
        ax.set_ylim(n + 0.1, -0.9)
        last = top - W + 1  # the W-th bit down from the vector's largest leading bit
        for i, v in enumerate(vals):
            track(ax, i, hi, lo)
            cells(ax, i, [(p, b, BLUE if p >= last else "#9a9a9a") for p, b in float_bits(v)], hi)
            if i in rs:
                ax.add_patch(Rectangle((0, i + 0.06), hi - lo + 1, 0.88, facecolor="none",
                                       edgecolor=GOLD, linewidth=1.2))
                lead = float_bits(v)[0][0]
                if lead < top:  # the target pair's small entry: ESC bits below the top
                    span_arrow(ax, col(top), col(lead), i + 0.62, rf"${top - lead}$", fontsize=7.5)
        ax.plot([col(last) + 1] * 2, [-0.3, n + 0.1], color="#222", linewidth=0.8, dashes=(2, 2))
        span_arrow(ax, col(top), col(last) + 1, -0.45, rf"$W = 53 + \mathrm{{ESC}} = {W}$")
        left = fig.add_subplot(grid[row, 0])
        contents_panel(left, vals, names=[f"{name}_{i + 1}" for i in range(n)])
        left.set_ylim(n + 0.1, -0.9)
        left.set_title(f"({'ab'[row]}) ${name}$", loc="left", fontsize=9, pad=4)

    # (c) the products' exponents on the same columns, larger to the left
    ax = fig.add_subplot(grid[2, 1])
    ax.set_xlim(0, hi - lo + 1)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    power_ticks(ax, hi, lo, 8)
    for e in sorted(set(ez), reverse=True):
        same = [i for i, ei in enumerate(ez) if ei == e]  # pairs with equal exponents share a dot
        ax.plot(col(e) + 0.5, 0, "o", color=GOLD if e == zr else BLUE, markersize=5)
        ax.text(col(e) + 0.5, 0.3, "$" + ", ".join(f"z_{i + 1}" for i in same) + "$",
                ha="center", va="bottom", fontsize=8)
    ax.plot(col(xp + yq) + 0.5, 0, "o", markerfacecolor="white", markeredgecolor="#222", markersize=5,
            clip_on=False)
    # edge to edge like the arrows in (a) and (b), so all three span the same columns
    span_arrow(ax, col(xp + yq), col(zr), 1.35, rf"$\mathrm{{ESC}} = {esc}$")
    left = fig.add_subplot(grid[2, 0])
    left.axis("off")
    left.text(0, 0.5, r"(c) $\exp(x_i) + \exp(y_i)$", ha="left", va="center", fontsize=9,
              transform=left.transAxes)
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in [
        ("addition_rounding", figure_addition), ("split_pi_M24", figure_split), ("span_slices", figure_span),
        ("split_int", figure_split_int), ("esc", figure_esc),
        ("slice_ozaki", figure_slice_ozaki), ("slice_immu", figure_slice_immu), ("slice_esc", figure_slice_esc),
        ("product_ozaki", lambda: figure_product_steps("ozaki")), ("product_immu", lambda: figure_product_steps("immu")),
        ("product_esc", lambda: figure_product_steps("esc")),
    ]:
        fig = make()
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        print("wrote", OUT / f"{name}.pdf")
