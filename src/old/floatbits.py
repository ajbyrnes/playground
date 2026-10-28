import math
import struct

def float_bits(x):
    """Split a float64 into its raw (sign, exponent, mantissa) fields."""
    bits = int.from_bytes(struct.pack(">d", x))

    sign = bits >> 63
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)
    return sign, exponent, mantissa


def significand_bits(x):
    """The 53 significand bits of x as (power_of_two, bit) pairs, leading bit first.

    For 100.0 = 1.1001b * 2^6 this gives (6, 1), (5, 1), (4, 0), (3, 0), (2, 1), ...
    down to (6 - 52, 0). Returns [] for 0; subnormals and inf/nan aren't handled.
    """
    if x == 0:
        return []

    sign, exponent, mantissa = float_bits(x)
    top = exponent - 1023                # power of two of the leading bit
    significand = (1 << 52) | mantissa   # put back the hidden leading 1

    return [(top - i, (significand >> (52 - i)) & 1) for i in range(53)]


# Color says which quantity a row is; fill says the bit: solid = 1, hatched = 0.
# Positions where a value has no bit show the grey track behind the row.
# A categorical palette checked for color-blind separation, used in this order.
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "red": "#e34948",
}
TRACK_COLOR = "#ececec"
HATCH = dict(shape="/", size=5, solidity=0.35, bgcolor="white")

CASES = {
    "[100, 10, 1, 0.1]: decade spread": [100.0, 10.0, 1.0, 0.1],
    "[8, 4, 2, 1, 0.5]: powers of two": [8.0, 4.0, 2.0, 1.0, 0.5],
    "[0.1, 0.2, 0.3]: repeating fractions": [0.1, 0.2, 0.3],
    "[1 + 2^-52, 1, 1 - 2^-53]: neighbours of 1": [1 + 2**-52, 1.0, 1 - 2**-53],
    "[1e16, 1, -1e16]: wide magnitude gap": [1e16, 1.0, -1e16],
}


def label_columns(name, x):
    """Row label for x split into columns, e.g. x[0], = 100.0, = 1.5625, × 2^6.

    The significand is in [1, 2), so the power of two is the position of the
    leading bit (the same exponent float_bits gives, minus the 1023 bias).
    """
    x = float(x)  # so NumPy scalars print as 0.1, not np.float64(0.1)
    if x == 0:
        return [name, f"= {x!r}", "", ""]
    sign, exponent, _ = float_bits(x)
    power = exponent - 1023
    significand = abs(x) / 2.0**power  # exact: scaling by a power of two only changes the exponent
    return [
        name,
        f"= {x!r}",
        f"= {'−' if sign else ''}{significand!r}",
        f"× 2<sup>{power}</sup>",
    ]


LABEL_FONT = dict(family="monospace", size=12)


def text_width(text):
    """Rough pixel width of text in LABEL_FONT (HTML tags don't render)."""
    import re

    return 7.5 * len(re.sub(r"<[^>]+>", "", text))


def column_widths(rows):
    """Pixel width of each label column: its widest entry across `rows`."""
    return [max(text_width(row[c]) for row in rows) for c in range(len(rows[0]))]


def label_annotations(values, row_names, widths=None, ys=None, label_rows=None):
    """Row labels as annotations laid out in columns, so the = and × signs line up.

    Each annotation sits at the left edge of the plot (x=0 in paper coordinates)
    and is pushed left into the margin by an exact number of pixels with xshift.
    Pass `widths` to keep the columns fixed across several plots (see column_widths),
    and `ys` if the rows were placed at numeric positions (see bit_grid).
    `label_rows` replaces the default label_columns text with your own columns.
    Returns the annotations and the margin width they need.
    """
    rows = label_rows or [label_columns(name, x) for name, x in zip(row_names, values)]
    gap = 10  # px between columns
    widths = widths or column_widths(rows)

    # left edge of each column, measured in px from the left edge of the figure
    starts = [gap + sum(widths[:c]) + gap * c for c in range(len(widths))]
    margin = starts[-1] + widths[-1] + 2 * gap

    annotations = [
        dict(
            text=text, x=0, y=y, xref="paper", yref="y",
            xanchor="left", xshift=start - margin,
            showarrow=False, font=LABEL_FONT,
        )
        for y, row in zip(ys or row_names, rows)
        for text, start in zip(row, starts)
        if text
    ]
    return annotations, margin


def bit_grid(values, names=None, powers=None, color_of=None, ys=None, bits_of=None):
    """Bar data for one case: every bit of every value, grouped by (color, bit).

    `names` defaults to x[0], x[1], ...; `powers` defaults to just the span the
    values cover (pass it to keep the bit space fixed across several plots).
    `color_of(row, power)` picks a color name from COLORS for each cell; by
    default everything is blue. `ys` places rows at numeric positions instead
    of one category per name, which allows gaps between groups of rows.
    `bits_of(value)` gives a value's (power, bit) pairs; the default is a
    double's 53 significand bits, but exact values can supply their own.

    The groups always come out in the same order (see trace_order), even when
    some are empty, so a dropdown or slider can swap data trace-by-trace.
    """
    names = names or [f"x[{i}]" for i in range(len(values))]
    ys = ys or names
    color_of = color_of or (lambda row, power: "blue")
    bits_of = bits_of or (lambda x: significand_bits(float(x)))
    rows = [dict(bits_of(x)) for x in values]  # {power: bit} per value
    if powers is None:
        all_powers = [p for row in rows for p in row]
        # high to low, plus one spare column on the left for sign markers
        powers = list(range(max(all_powers) + 1, min(all_powers) - 1, -1))

    groups = {key: dict(x=[], base=[], y=[], hovertext=[]) for key in trace_order()}
    for i, (name, y, x, row) in enumerate(zip(names, ys, values, rows)):
        for p, bit in row.items():
            group = groups[(color_of(i, p), bit)]
            group["x"].append(1)          # each bar is one bit position wide...
            group["base"].append(p - 0.5)  # ...centered on its power of two
            group["y"].append(y)
            # kept short: there's one per bar, and the row label already shows the value
            group["hovertext"].append(f"{name}: bit at 2^{p} = {bit}")

    # the grey track: one bar per row spanning the whole bit space
    track = dict(
        x=[len(powers)] * len(names), base=[powers[-1] - 0.5] * len(names),
        y=ys, hovertext=[f"{name} ≈ {float(x)!r}" for name, x in zip(names, values)],
    )
    return dict(
        powers=powers, names=names, ys=ys, traces=[track] + list(groups.values()),
        signs=sign_markers(values, rows, ys),
    )


SIGN_FONT = dict(family="monospace", size=13, color="#333")


def sign_markers(values, rows, ys):
    """A + or − just left of each nonzero row's leading bit, as annotations.

    The bars only show magnitude (that's how floats store numbers: a sign bit
    plus a magnitude), so this puts the sign where the eye already is, and
    lets you see +a and −a line up when they cancel.
    """
    markers = []
    for x, row, y in zip(values, rows, ys):
        if not row:
            continue  # zero: no bits, no sign to show
        top = max(row)
        negative = x < 0 or (x == 0 and math.copysign(1, x) < 0)
        markers.append(dict(
            text="−" if negative else "+", x=top + 0.5, y=y, xref="x", yref="y",
            xanchor="right", xshift=-1, showarrow=False, font=SIGN_FONT,
        ))
    return markers


def trace_order():
    """The (color, bit) groups in bit_grid's traces, after the track."""
    return [(color, bit) for color in COLORS for bit in (1, 0)]


def trace_update(grid):
    """Arguments that swap a figure's bars for this grid's: (data changes, trace indices).

    Used by "update" buttons and slider steps. Each data key maps to a list
    with one entry per targeted trace.
    """
    keys = ("x", "base", "y", "hovertext")
    data = {key: [trace[key] for trace in grid["traces"]] for key in keys}
    return data, list(range(len(grid["traces"])))


def case_layout(values, grid):
    """Layout settings that depend on the case, as plotly's dotted paths.

    Labels every 4th column with its power of two (e.g. 2⁻⁴; hover gives the
    rest), lays out the row labels, sizes the figure to the case, and titles it
    with the width of the common bit space.
    """
    powers = grid["powers"]
    ticks = [p for p in powers if p % 4 == 0]
    annotations, left = label_annotations(values, grid["names"])
    return {
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{p}</sup>" for p in ticks],
        "annotations": annotations + grid["signs"],
        "margin.l": left,
        "width": left + 20 + 12 * len(powers),  # 12px per column: zoomed out enough to see each float's window
        "height": 180 + 40 * len(values),
        "title.text": (
            f"Common bit space: 2<sup>{powers[0]}</sup> to 2<sup>{powers[-1]}</sup> "
            f"= {len(powers)} positions · each float holds 53"
        ),
    }


SECTION_COLOR = "#555"


def section_marks(sections, row_ys, left, plot_width):
    """Section headers and dividers for a plot whose rows are grouped into sections.

    `sections` is a list of (y of the section's first row, header text). Each
    header sits just above its first row, starting at the figure's left edge;
    each section after the first gets a divider halfway between it and the row
    above, running across the whole figure (paper x < 0 is the label margin).
    Returns (annotations, shapes).
    """
    annotations, shapes = [], []
    for first_y, text in sections:
        annotations.append(dict(
            text=f"<b>{text}</b>", x=0, y=first_y - 0.55, xref="paper", yref="y",
            xanchor="left", yanchor="bottom", xshift=10 - left, showarrow=False,
            font=dict(color=SECTION_COLOR, size=11),
        ))
        above = [y for y in row_ys if y < first_y]
        if above:
            y = (max(above) + first_y) / 2
            shapes.append(dict(
                type="line", xref="paper", yref="y", x0=-left / plot_width, x1=1,
                y0=y, y1=y, line=dict(color="#bbb", width=1),
            ))
    return annotations, shapes


def base_figure(grid):
    """The bit bars for one grid, with the track, legend and axis styling.

    Case-dependent layout (ticks, labels, size, title) is left to case_layout.
    """
    import plotly.graph_objects as go

    track, *groups = grid["traces"]
    fig = go.Figure(go.Bar(
        **track, orientation="h", width=0.7, hoverinfo="text",
        marker=dict(color=TRACK_COLOR), showlegend=False,
    ))
    for (color, bit), group in zip(trace_order(), groups):
        fig.add_trace(go.Bar(
            **group, orientation="h", width=0.7, hoverinfo="text",
            marker=dict(
                color="white" if bit == 0 else COLORS[color],
                pattern=dict(HATCH, fgcolor=COLORS[color]) if bit == 0 else None,
                line=dict(color="white", width=1),  # thin gaps between bits
            ),
            showlegend=False,
        ))

    # The legend explains the fills; the row labels say what each color is.
    for name, pattern in [("1 bit", None), ("0 bit", dict(HATCH, fgcolor="#555")), ("no bit", None)]:
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h", name=name,
            marker=dict(
                color={"1 bit": "#555", "0 bit": "white", "no bit": TRACK_COLOR}[name],
                pattern=pattern, line=dict(color="#999", width=1),
            ),
        ))
    fig.update_xaxes(
        autorange="reversed",  # big powers on the left
        title="bit position",
        tickmode="array",
        tickangle=0,
        tickfont=dict(size=10),
        showgrid=False, zeroline=False,
    )
    fig.update_yaxes(
        autorange="reversed",  # first value on top
        showgrid=False,
        showticklabels=False,  # label_annotations draws the row labels instead
    )
    fig.update_layout(
        plot_bgcolor="white",
        barmode="overlay",  # bars in a row sit side by side via `base`, not stacked
        margin=dict(r=20, t=110, b=50),
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top"),
    )
    return fig


def m_slider(grids, layouts, Ms, start_M):
    """A slider over M whose steps swap in each M's grid and layout (same "update" as a dropdown)."""
    steps = []
    for M in Ms:
        data, indices = trace_update(grids[M])
        steps.append(dict(label=str(M), method="update", args=[data, layouts[M], indices]))
    return dict(active=Ms.index(start_M), currentvalue=dict(prefix="M = "), pad=dict(t=50), steps=steps)


def slider_figure(views):
    """A figure with an M slider and, for several cases, a dropdown to pick the case.

    `views` maps a dropdown label to (grid, layout, slider) for that case. Plotly
    controls can't read each other's state, so each dropdown entry swaps in that
    case's own slider along with its plot; picking a new case resets M.
    """
    grid, layout, slider = next(iter(views.values()))
    fig = base_figure(grid)
    # update_layout also accepts "xaxis_tickvals" for xaxis.tickvals, so swap the dots
    fig.update_layout(**{key.replace(".", "_"): value for key, value in layout.items()})
    fig.update_layout(margin_b=110, sliders=[slider])  # room for the slider under the x-axis

    if len(views) > 1:
        buttons = []
        for label, (grid, layout, slider) in views.items():
            data, indices = trace_update(grid)
            buttons.append(dict(
                label=label, method="update",
                args=[data, {**layout, "sliders": [slider]}, indices],
            ))
        fig.update_layout(updatemenus=[dict(
            buttons=buttons, direction="down",
            x=1, xanchor="right", y=1.12, yanchor="bottom",  # top right, clear of labels on the plot's top edge
            showactive=True,
        )])
    return fig


def plot_floats(cases):
    """Draw each float as a row of its 53 significand bits, aligned by place value.

    `cases` maps a name to a list of floats; with more than one case, a dropdown
    switches between them. Bits further left are worth more, so numbers of
    different size are shifted against each other, the way they actually line up
    when you add them. Hover over a cell to see its bit and power of two.
    """
    grids = {name: bit_grid(values) for name, values in cases.items()}
    first = next(iter(grids.values()))
    fig = base_figure(first)

    # update_layout also accepts "xaxis_tickvals" for xaxis.tickvals, so swap the dots
    first_layout = case_layout(next(iter(cases.values())), first)
    fig.update_layout(**{key.replace(".", "_"): value for key, value in first_layout.items()})

    if len(grids) > 1:
        # Each button swaps in one case's grid. method="update" takes the trace
        # changes, the layout changes, and which traces to apply the first to.
        buttons = []
        for name, grid in grids.items():
            data, indices = trace_update(grid)
            buttons.append(dict(
                label=name, method="update",
                args=[data, case_layout(cases[name], grid), indices],
            ))
        fig.update_layout(updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0, xanchor="left", y=1.02, yanchor="bottom",  # just above the plot
            showactive=True,
        )])
    return fig


if __name__ == "__main__":
    for x in [1.0, 0.1, -2.5, 100.0]:
        sign, exponent, mantissa = float_bits(x)
        print(f"{x:>6}: sign={sign} exponent={exponent} (2^{exponent - 1023}) mantissa={mantissa:052b}")

    fig = plot_floats(CASES)
    fig.write_html("floatbits.html")
