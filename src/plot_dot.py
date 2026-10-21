from fractions import Fraction

from floatbits import (
    base_figure, bit_grid, column_widths, label_annotations, section_marks, significand_bits,
    trace_update,
)
from products import DOT_CASES, dot_info, padded_bits, significant_bits, split_dot

GAP = 1.2

# Intermediate values (products, running sums) are shown exactly, before
# rounding: aqua up to their 53rd bit, red past it, because that's what rounding drops. Values computed in floating point are
# blue, and the exact answer they're aiming for is magenta.
TERM_COLOR, LOST_COLOR, COMPUTED_COLOR, EXACT_COLOR = "aqua", "red", "blue", "magenta"


def bits_of(v):
    """Doubles show all 53 significand bits; exact values too, or more if they need them."""
    return significand_bits(v) if isinstance(v, float) else padded_bits(v)


def exact_product(a, b):
    return Fraction(float(a)) * Fraction(float(b))


def bits_text(count):
    return f"{count} bit" if count == 1 else f"{count} bits"


def dot_rows(xs, ys):
    """Every row of the plot for one example, as sections of (name, value, kind, label columns).

    kind is "term" (an exact intermediate value that then gets rounded to a
    double), "computed" (a double), or "exact" (the true answer).
    """
    plain = dot_info(xs, ys)
    split = split_dot(xs, ys)
    exact = plain["exact"]

    def error_text(computed, target):
        error = Fraction(computed) - target
        if error == 0:
            return "no error"
        if target == 0:
            return f"error {float(error):.2g}"
        return f"rel. error {float(abs(error / target)):.2g}"

    # label columns: name, what was combined, value, bit count, note
    def term_row(name, operands, value):
        bits = significant_bits(value)
        fit = "fits" if bits <= 53 else f"{bits - 53} don't fit"
        return (name, value, "term", [name, operands, f"≈ {float(value):.6g}", bits_text(bits), fit])

    # operands get 10 digits, enough to see they aren't the values they look like
    def sum_text(a, b):
        return f"{a:.10g} − {-b:.10g}" if b < 0 else f"{a:.10g} + {b:.10g}"

    def computed_row(name, value, target):
        return (name, value, "computed", [name, "", f"= {value:.6g}", "", error_text(value, target)])

    x_hi, x_lo, y_hi, y_lo = split["x_hi"], split["x_lo"], split["y_hi"], split["y_lo"]
    n = len(xs)
    # plain: each running sum, exactly as it is before rounding (the products
    # themselves are what products.html is for)
    plain_rows = []
    total = float(xs[0]) * float(ys[0])
    for i in range(1, n):
        product = float(xs[i]) * float(ys[i])
        # the partial sum through term i, added from the already-rounded sum so far
        name = "x[0]·y[0] + x[1]·y[1]" if i == 1 else f"x[0]·y[0] + … + x[{i}]·y[{i}]"
        plain_rows.append(term_row(name, sum_text(total, product), Fraction(total) + Fraction(product)))
        total += product

    split_rows = []
    targets = [
        sum(exact_product(a, b) for a, b in zip(x_hi, y_hi)),
        sum(exact_product(a, b) for a, b in zip(x_hi, y_lo)),
        sum(exact_product(a, b) for a, b in zip(x_lo, ys)),
    ]
    for name, value, target in zip(["Σ x_hi·y_hi", "Σ x_hi·y_lo", "Σ x_lo·y"], split["parts"], targets):
        split_rows.append(computed_row(name, value, target))

    # the two answers side by side with the true value they're aiming for
    result_rows = [
        computed_row("plain result", plain["rounded"], exact),
        computed_row("split result", split["total"], exact),
        ("exact answer", exact, "exact",
         ["exact answer", "", f"≈ {float(exact):.6g}", bits_text(significant_bits(exact)), ""]),
    ]

    sections = [
        ("PLAIN: each running sum, before it's rounded to 53 bits", plain_rows),
        (f"SPLIT (M = {split['M']}): xᵀy = x_hiᵀy_hi + x_hiᵀy_lo + x_loᵀy, and the first part is computed exactly",
         split_rows),
        ("RESULTS, compared with the exact answer (computed with exact fractions, never rounded)",
         result_rows),
    ]
    return sections, plain, split


def color_for(rows):
    """Colors by kind: terms aqua up to their 53rd bit and red past it."""
    tops = [bits[0][0] if (bits := bits_of(value)) else 0 for _, value, _, _ in rows]

    def color_of(row, power):
        kind = rows[row][2]
        if kind == "term":
            return TERM_COLOR if power > tops[row] - 53 else LOST_COLOR
        return COMPUTED_COLOR if kind == "computed" else EXACT_COLOR

    return color_of


def dot_view(name, xs, ys):
    """Grid and layout for one example."""
    sections, plain, split = dot_rows(xs, ys)

    # rows numbered down the page, with a GAP between sections
    rows, row_ys, headers = [], [], []
    y = 0.0
    for header, section_rows in sections:
        headers.append((y, header))
        for row in section_rows:
            rows.append(row)
            row_ys.append(y)
            y += 1
        y += GAP

    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    labels = [r[3] for r in rows]
    grid = bit_grid(values, names, None, color_for(rows), row_ys, bits_of)

    annotations, left = label_annotations(None, names, column_widths(labels), row_ys, labels)
    plot_width = 8 * len(grid["powers"])
    section_headers, dividers = section_marks(headers, row_ys, left, plot_width)

    def rel(v):
        error = abs(Fraction(v) - plain["exact"])
        return f"{float(error / abs(plain['exact'])):.2g}" if plain["exact"] else f"{float(error):.2g}"

    powers = grid["powers"]
    ticks = [q for q in powers if q % 4 == 0]
    layout = {
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{q}</sup>" for q in ticks],
        "annotations": annotations + section_headers + grid["signs"],
        "shapes": dividers,
        "yaxis.range": [row_ys[-1] + 0.6, row_ys[0] - 1.0],  # reversed: first row on top
        "yaxis.autorange": False,
        "margin.l": left,
        "width": left + 20 + plot_width,
        "height": 160 + 40 * (row_ys[-1] + 1),
        "title.text": (
            f"x = [{', '.join(repr(float(v)) for v in xs)}]    "
            f"y = [{', '.join(repr(float(v)) for v in ys)}]<br>"
            f"<span style='font-size:12px'>plain relative error {rel(plain['rounded'])}  ·  "
            f"split relative error {rel(split['total'])}</span>"
        ),
    }
    return grid, layout


def plot_dot(cases=DOT_CASES):
    """Plain vs split (equation (7)) dot products for each example, with a dropdown."""
    views = {name: dot_view(name, xs, ys) for name, (xs, ys) in cases.items()}

    grid, layout = next(iter(views.values()))
    fig = base_figure(grid)
    fig.update_layout(**{key.replace(".", "_"): value for key, value in layout.items()})

    if len(views) > 1:
        buttons = []
        for name, (grid, layout) in views.items():
            data, indices = trace_update(grid)
            buttons.append(dict(label=name, method="update", args=[data, layout, indices]))
        fig.update_layout(updatemenus=[dict(
            buttons=buttons, direction="down",
            x=1, xanchor="right", y=1.02, yanchor="bottom", showactive=True,
        )])
    return fig


if __name__ == "__main__":
    plot_dot().write_html("dot.html")
