import math
from fractions import Fraction

from floatbits import (
    bit_grid, column_widths, label_annotations, m_slider, section_marks, significand_bits,
    slider_figure,
)
from products import padded_bits, product_info, significant_bits, split_products

PLAIN = ["x", "y", "x·y exact", "fl(x·y)"]
PIECES = ["x_hi·y_hi", "x_hi·y_lo", "x_lo·y_hi", "x_lo·y_lo"]
NAMES = PLAIN + PIECES
EXACT_ROWS = {"x·y exact", *PIECES}

GAP = 1.2
YS = [0, 1, 2, 3] + [4 + GAP + i for i in range(4)]
SECTIONS = [
    (YS[0], "PLAIN PRODUCT: x·y needs up to 106 bits, but a double holds 53"),
    (YS[4], "SPLIT PRODUCTS: multiply the ExtractScalar pieces of x and y instead"),
]

# x and y keep their own colors; every product is aqua, except the bits a
# double can't hold (past the 53rd), which are red because rounding drops them
ROW_COLORS = {"x": "blue", "y": "orange"}
PRODUCT_COLOR, LOST_COLOR = "aqua", "red"


def bits_of(v):
    """Doubles show all 53 significand bits; exact products too, or more if they need them."""
    return significand_bits(v) if isinstance(v, float) else padded_bits(v)


def product_rows(x, y, M):
    """Values and label columns for every row at one M."""
    plain = product_info(x, y)
    pieces = split_products(x, y, M)
    values = [x, y, Fraction(x) * Fraction(y), plain["rounded"]]
    values += [Fraction(p["a"]) * Fraction(p["b"]) for p in pieces.values()]

    def error_text(error):
        return "exact" if error == 0 else f"error {float(error):.2g}"

    labels = [
        ["x", f"= {x!r}", f"{significant_bits(x)} bits", ""],
        ["y", f"= {y!r}", f"{significant_bits(y)} bits", ""],
        ["x·y exact", f"≈ {float(values[2])!r}", f"{plain['exact_bits']} bits",
         f"{max(plain['exact_bits'] - 53, 0)} don't fit"],
        ["fl(x·y)", f"= {plain['rounded']!r}", "53 bits", error_text(plain["error"])],
    ]
    for name, p in pieces.items():
        labels.append([
            name, f"≈ {float(Fraction(p['a']) * Fraction(p['b'])):.6g}",
            f"{significant_bits(p['a'])} × {significant_bits(p['b'])} → {p['exact_bits']} bits",
            error_text(p["error"]),
        ])
    rounded_pieces = sum(p["error"] != 0 for p in pieces.values())
    return values, labels, rounded_pieces


def color_for(values):
    """Row colors: products are aqua up to their 53rd bit and red past it."""
    tops = [bits[0][0] if (bits := bits_of(v)) else 0 for v in values]

    def color_of(row, power):
        name = NAMES[row]
        if name in EXACT_ROWS:
            return PRODUCT_COLOR if power > tops[row] - 53 else LOST_COLOR
        return ROW_COLORS.get(name, PRODUCT_COLOR)

    return color_of


def step_layout(x, y, M, labels, rounded_pieces, grid, widths):
    """Layout for one slider step: row labels, section headers, divider, title."""
    annotations, left = label_annotations(None, NAMES, widths, YS, labels)
    plot_width = 8 * len(grid["powers"])

    headers, dividers = section_marks(SECTIONS, YS, left, plot_width)
    annotations += headers

    verdict = ("all four split products are exact" if rounded_pieces == 0
               else f"{rounded_pieces} of 4 split products still rounded")
    powers = grid["powers"]
    ticks = [q for q in powers if q % 4 == 0]
    return {
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{q}</sup>" for q in ticks],
        "annotations": annotations + grid["signs"],
        "shapes": dividers,
        "yaxis.range": [YS[-1] + 0.6, YS[0] - 1.0],  # reversed: first row on top, room for a header
        "yaxis.autorange": False,
        "margin.l": left,
        "width": left + 20 + plot_width,
        "height": 300 + 40 * len(NAMES),
        "title.text": f"x = {x!r}, y = {y!r}, M = {M}: {verdict}",
    }


def product_views(x, y, Ms, start_M):
    """Start grid, start layout, and M slider for one (x, y) pair; same idea as plot_split.split_views."""
    steps = {M: product_rows(x, y, M) for M in Ms}

    # one bit space and one set of label columns for every M
    all_powers = [q for values, _, _ in steps.values() for v in values for q, _ in bits_of(v)]
    powers = list(range(max(all_powers) + 1, min(all_powers) - 1, -1))  # spare column for sign markers
    widths = column_widths([row for _, labels, _ in steps.values() for row in labels])

    grids, layouts = {}, {}
    for M, (values, labels, rounded_pieces) in steps.items():
        grids[M] = bit_grid(values, NAMES, powers, color_for(values), YS, bits_of)
        layouts[M] = step_layout(x, y, M, labels, rounded_pieces, grids[M], widths)

    return grids[start_M], layouts[start_M], m_slider(grids, layouts, Ms, start_M)


def plot_products(pairs, Ms=range(0, 54), start_M=27):
    """Plain x·y vs the products of ExtractScalar pieces, with an M slider and a pair dropdown.

    Picking a new pair resets M to start_M (see floatbits.slider_figure).
    """
    Ms = list(Ms)
    return slider_figure({
        f"x = {x:.6g}, y = {y:.6g}": product_views(x, y, Ms, start_M) for x, y in pairs
    })


if __name__ == "__main__":
    fig = plot_products([(0.1, 1 / 3), (math.pi, math.e), (1.1, 0.9)])
    fig.write_html("products.html")
