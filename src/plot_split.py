import math

import numpy as np

from floatbits import (
    bit_grid, column_widths, float_bits, label_annotations,
    label_columns, m_slider, section_marks, significand_bits, slider_figure,
)
from splitting import extract_scalar

NAMES = ["σ", "x", "σ + x", "x_hi", "x_lo"]
CUT_COLOR = "#222"

# Two sections: the mechanics of the split (rows 0-2), then its output (rows 3-4),
# with an extra GAP between them for a divider and the section headers.
GAP = 1.2
YS = [0, 1, 2, 3 + GAP, 4 + GAP]
SECTIONS = [
    (YS[0], "SPLIT MECHANICS: σ + x drops every bit of x below the cut"),
    (YS[3], "OUTPUT: x = x_hi + x_lo, exactly"),
]

# one color per quantity, in the palette's order
ROW_COLORS = dict(zip(NAMES, ["blue", "orange", "aqua", "yellow", "magenta"]))


def sigma_for(x, M):
    """σ from equation (1), computed the same way extract_scalar does."""
    return float(2**M * 2**(np.ceil(np.log2(np.abs(x)))))


def split_rows(x, M):
    """The five values shown for one M: σ, x, fl(σ + x), and the split x_hi + x_lo."""
    sigma = sigma_for(x, M)
    x_hi, x_lo = extract_scalar(x, M)
    return [sigma, x, sigma + x, float(x_hi), float(x_lo)]


def leading_power(x):
    """Power of two of x's leading bit."""
    return float_bits(x)[1] - 1023


def cut_position(rows):
    """Power of two of the lowest bit kept in σ + x; every bit of x below it is rounded away."""
    return leading_power(rows[NAMES.index("σ + x")]) - 52


def bits_each_side(x, cut):
    """How many of x's 53 bit positions are at or above the cut (x_hi's side) and below it (x_lo's)."""
    top = leading_power(x)
    hi = min(max(top - cut + 1, 0), 53)
    return hi, 53 - hi


def step_layout(x, M, rows, grid, widths):
    """Layout for one slider step: row labels, the cut line, and a title."""
    annotations, left = label_annotations(rows, NAMES, widths, YS)
    cut = cut_position(rows)

    # the cut line sits between the last kept position and the first dropped
    # one, and runs through both sections: it's one position in the shared space
    cut_line = dict(
        type="line", xref="x", yref="paper", x0=cut - 0.5, x1=cut - 0.5, y0=0, y1=1,
        line=dict(color=CUT_COLOR, width=2, dash="dot"),
    )
    hi_bits, lo_bits = bits_each_side(x, cut)
    annotations.append(dict(
        text=f"x_hi ← {hi_bits} bits ┊ {lo_bits} bits → x_lo",
        x=cut - 0.5, y=1, xref="x", yref="paper",
        xanchor="center", yanchor="bottom", showarrow=False, font=dict(color=CUT_COLOR, size=11),
    ))

    plot_width = 8 * len(grid["powers"])
    headers, dividers = section_marks(SECTIONS, YS, left, plot_width)
    annotations += headers
    powers = grid["powers"]
    ticks = [q for q in powers if q % 4 == 0]
    return {
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{q}</sup>" for q in ticks],
        "annotations": annotations + grid["signs"],
        "shapes": [cut_line, *dividers],
        "yaxis.range": [YS[-1] + 0.6, YS[0] - 1.0],  # reversed: first row on top, room for a header
        "yaxis.autorange": False,  # use the range above, not base_figure's autorange
        "margin.l": left,
        "width": left + 20 + plot_width,  # 8px per column, narrower than plot_floats: the fixed space is wide
        "height": 300 + 40 * len(rows),  # includes room for the slider and section headers
        "title.text": (
            f"ExtractScalar(x = {x!r}, M = {M}):  "
            f"σ = 2<sup>{M}</sup> · 2<sup>⌈log₂|x|⌉</sup> = 2<sup>{leading_power(rows[0])}</sup>"
        ),
    }


def split_views(x, Ms, start_M):
    """Everything plot_extract_scalar shows for one x: the start grid and layout, and its M slider.

    The bit space is fixed across all M so you can watch σ move left as M
    grows, dragging the cut with it.
    """
    steps = {M: split_rows(x, M) for M in Ms}

    # one bit space and one set of label columns for every step, so nothing
    # jumps around while dragging the slider
    all_powers = [q for rows in steps.values() for v in rows for q, _ in significand_bits(v)]
    powers = list(range(max(all_powers) + 1, min(all_powers) - 1, -1))  # spare column for sign markers
    widths = column_widths([label_columns(n, v) for rows in steps.values() for n, v in zip(NAMES, rows)])

    grids = {M: bit_grid(rows, NAMES, powers, lambda row, power: ROW_COLORS[NAMES[row]], YS) for M, rows in steps.items()}
    layouts = {M: step_layout(x, M, steps[M], grids[M], widths) for M in Ms}

    return grids[start_M], layouts[start_M], m_slider(grids, layouts, Ms, start_M)


def plot_extract_scalar(xs, Ms=range(0, 56), start_M=20):
    """Algorithm 1 with a slider for M and, for several xs, a dropdown for x.

    x_hi keeps x's bits above the cut (rounded to nearest, so x_lo can be
    negative), x_lo gets the rest.

    Picking a new x resets M to start_M (see floatbits.slider_figure).
    """
    xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
    Ms = list(Ms)
    return slider_figure({f"x = {x!r}": split_views(x, Ms, start_M) for x in xs})


if __name__ == "__main__":
    fig = plot_extract_scalar([0.1, -0.1, 1 / 3, 100.0, math.pi, 1.0])
    fig.write_html("split.html")
