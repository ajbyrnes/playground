import numpy as np

from floatbits import (
    CASES, bit_grid, column_widths, float_bits, label_annotations, m_slider, section_marks,
    significand_bits, slider_figure,
)
from splitting import extract_vector

CUT_COLOR = "#222"
GAP = 1.2

# σ gets its own color; each element's bits are blue on x_hi's side of the
# cut and orange on x_lo's side
SIGMA_COLOR, HI_COLOR, LO_COLOR = "aqua", "blue", "orange"


def vector_sigma(xs, M):
    """σ for ExtractVector, computed the same way extract_vector does: from max|x|."""
    return float(2**M * 2**(np.ceil(np.log2(np.max(np.abs(xs))))))


def leading_power(v):
    """Power of two of v's leading bit."""
    return float_bits(v)[1] - 1023


def element_cut(sigma, x):
    """Lowest power of two x_hi can keep for this element: the bottom of fl(σ + x)'s window.

    It's the same for every positive element; for a negative one σ + x falls
    below σ, so its window (and its cut) is one position lower.
    """
    return leading_power(sigma + x) - 52


def bits_each_side(x, cut):
    """How many of x's 53 bit positions are at or above its cut (x_hi's side) and below it (x_lo's)."""
    if x == 0:
        return 0, 0
    hi = min(max(leading_power(x) - cut + 1, 0), 53)
    return hi, 53 - hi


def vector_rows(xs, M):
    """Values, label columns, and per-row cuts for one M: σ first, then each element."""
    sigma = vector_sigma(xs, M)
    x_hi, x_lo = extract_vector(np.asarray(xs, dtype=float), M)
    biggest = max(abs(v) for v in xs)

    values = [sigma] + list(xs)
    cuts = [None] + [element_cut(sigma, x) for x in xs]
    labels = [["σ", f"= 2<sup>{leading_power(sigma)}</sup>", "", "", ""]]
    for i, (x, hi, lo, cut) in enumerate(zip(xs, x_hi, x_lo, cuts[1:])):
        n_hi, n_lo = bits_each_side(x, cut)
        note = f"{n_hi} | {n_lo} bits" + ("  ← max" if abs(x) == biggest else "")
        labels.append([f"x[{i}]", f"= {x!r}", f"x_hi = {float(hi)!r}", f"x_lo = {float(lo)!r}", note])
    return values, labels, cuts, sigma


def color_for(cuts):
    def color_of(row, power):
        if row == 0:
            return SIGMA_COLOR
        return HI_COLOR if power >= cuts[row] else LO_COLOR

    return color_of


def step_layout(xs, M, sigma, grid, labels, widths, ys):
    """Layout for one M: row labels, sign markers, the cut line, sections, and a title."""
    annotations, left = label_annotations(None, grid["names"], widths, ys, labels)
    plot_width = 8 * len(grid["powers"])
    headers, dividers = section_marks([
        (ys[0], "σ COMES FROM THE LARGEST ELEMENT: σ = 2<sup>M</sup> · 2<sup>⌈log₂ max|x|⌉</sup>"),
        (ys[1], "EVERY ELEMENT IS SPLIT AT THE SAME CUT: blue bits go to x_hi, orange bits to x_lo"),
    ], ys, left, plot_width)

    # the cut for positive elements; a negative element's cut is one position
    # lower (see element_cut), which shows as one extra blue cell past the line
    cut = leading_power(sigma) - 52
    cut_line = dict(
        type="line", xref="x", yref="paper", x0=cut - 0.5, x1=cut - 0.5, y0=0, y1=1,
        line=dict(color=CUT_COLOR, width=2, dash="dot"),
    )
    cut_label = dict(
        text="x_hi ← ┊ → x_lo", x=cut - 0.5, y=1, xref="x", yref="paper",
        xanchor="center", yanchor="bottom", showarrow=False, font=dict(color=CUT_COLOR, size=11),
    )

    biggest = max(abs(v) for v in xs)
    powers = grid["powers"]
    ticks = [q for q in powers if q % 4 == 0]
    return {
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{q}</sup>" for q in ticks],
        "annotations": annotations + headers + grid["signs"] + [cut_label],
        "shapes": [cut_line, *dividers],
        "yaxis.range": [ys[-1] + 0.6, ys[0] - 1.0],  # reversed: first row on top, room for a header
        "yaxis.autorange": False,
        "margin.l": left,
        "width": left + 20 + plot_width,
        "height": 300 + 40 * len(ys),
        "title.text": (
            f"ExtractVector, M = {M}: max|x| = {biggest!r}, so "
            f"σ = 2<sup>{M}</sup> · 2<sup>{leading_power(sigma) - M}</sup> = 2<sup>{leading_power(sigma)}</sup>"
        ),
    }


def vector_views(xs, Ms, start_M):
    """Start grid, start layout, and M slider for one vector."""
    steps = {M: vector_rows(xs, M) for M in Ms}
    names = ["σ"] + [f"x[{i}]" for i in range(len(xs))]
    ys = [0] + [1 + GAP + i for i in range(len(xs))]

    # one bit space and one set of label columns for every M, plus a spare
    # column on the left for sign markers
    all_powers = [q for values, _, _, _ in steps.values() for v in values for q, _ in significand_bits(v)]
    powers = list(range(max(all_powers) + 1, min(all_powers) - 1, -1))
    widths = column_widths([row for _, labels, _, _ in steps.values() for row in labels])

    grids, layouts = {}, {}
    for M, (values, labels, cuts, sigma) in steps.items():
        grids[M] = bit_grid(values, names, powers, color_for(cuts), ys)
        layouts[M] = step_layout(xs, M, sigma, grids[M], labels, widths, ys)
    return grids[start_M], layouts[start_M], m_slider(grids, layouts, Ms, start_M)


def plot_extract_vector(cases=CASES, Ms=range(0, 54), start_M=20):
    """Algorithm 2 on whole vectors: one σ from max|x|, one cut applied to every element.

    M slider, plus a dropdown over `cases` (name -> list of floats).
    """
    Ms = list(Ms)
    return slider_figure({name: vector_views(xs, Ms, start_M) for name, xs in cases.items()})


if __name__ == "__main__":
    plot_extract_vector().write_html("vector.html")
