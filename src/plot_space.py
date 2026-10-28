"""One float64, shown twice: where it sits in floating-point space, and its bits.

Left: floating-point space itself, as a line of points -- the quantized reals.
Inside one region [2^e, 2^(e+1)) the representable values are evenly spaced, and
that spacing doubles every time you cross a power of two, so on a linear value
axis the points visibly thin out as you walk up the line and crowd together as
you walk down it. The regions are exactly the exponents: one region per e. This
view has no notion of bit-width -- it only cares where the quantization lands.

Right: the same number's actual bits, in the usual layout (solid = 1, hatched =
0), indexed from its leading 1.

The slider multiplies the value by powers of two, which is what moving the
exponent does: the bits on the right never change, while on the left the point
hops from region to region and the spacing around it doubles or halves.

The line is drawn at a deliberately coarse display precision (`bits`, default 4
significand bits) because binary64 packs 2^52 points into every region and you
cannot draw that. The structure is what's faithful: equal spacing within a
region, doubling across regions. The highlighted marker is the real value, at
its real position, and the title carries its real ulp.
"""

import math
import struct

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from floatbits import COLORS, HATCH, TRACK_COLOR

BIAS = 1023
MB = 52        # stored significand bits
SB = 53        # significand bits, counting the hidden leading 1
PMIN = -1074   # the smallest place value binary64 has
EMAX = 1023
EDRAW = EMAX - 1   # highest region whose upper edge 2^(e+1) is still a float
ESUB = -1022   # the exponent of every subnormal

# Blue says "significand bit", orange says "the leading 1 the exponent points
# at" -- the same bit that is the highlighted point on the left.
BIT_COLOR, LEAD_COLOR = COLORS["blue"], COLORS["orange"]
VALUE_COLOR = COLORS["red"]      # "you are here" on the line
POINT_COLOR = "#b9c0c9"
EDGE_COLOR = "#777"
REGION_FILL = "rgba(42,120,214,0.055)"

VALUES = {
    "100": 100.0,
    "1": 1.0,
    "0.1": 0.1,
    "pi": math.pi,
    "1 + 2^-52": 1.0 + 2.0**-52,
    "1e16": 1e16,
    "2^-1022: min normal": 2.0**-1022,
}


# --------------------------------------------------------------- the number

def decode(x):
    """The raw (sign, exponent field, mantissa) of a float64."""
    bits = int.from_bytes(struct.pack(">d", float(x)))
    return bits >> 63, (bits >> 52) & 0x7FF, bits & ((1 << MB) - 1)


def parts(x):
    """x as sign, integer significand M and exponent q: value = (-1)^sign * M * 2^q.

    `lead` is the power of two of the leading set bit -- the region x falls in --
    and `bot` the lowest place its bits reach: 2^(lead-52) for a normal, and the
    hard 2^-1074 floor once x is subnormal, where the spacing stops halving and
    stays put all the way down to zero.
    """
    sign, ef, man = decode(x)
    if ef == 0x7FF:
        return dict(kind="inf" if man == 0 else "nan", sign=sign)
    if ef == 0:
        if man == 0:
            return dict(kind="zero", sign=sign, M=0, q=PMIN, lead=-1023, bot=PMIN)
        return dict(kind="subnormal", sign=sign, M=man, q=PMIN,
                    lead=PMIN + man.bit_length() - 1, bot=PMIN)
    e = ef - BIAS
    return dict(kind="normal", sign=sign, M=(1 << MB) | man, q=e - MB, lead=e, bot=e - MB)


def pack(sign, M, q):
    """(-1)^sign * M * 2^q back into a float64, truncated toward zero if it no longer fits.

    Returns (x, dropped). `dropped` counts the low places discarded, and stays 0
    when they were all zero, so it only reports precision that was really lost.
    """
    if M == 0:
        return (-0.0 if sign else 0.0), 0
    power = q + M.bit_length() - 1          # the leading bit's place value
    if power > EMAX:
        return (-math.inf if sign else math.inf), 0

    # a normal keeps 53 significand bits; a subnormal instead pins q at -1074
    shift = (M.bit_length() - SB) if power >= ESUB else (PMIN - q)
    dropped = 0
    if shift > 0:
        dropped = shift if M & ((1 << shift) - 1) else 0
        M >>= shift
    elif shift < 0:
        M <<= -shift

    ef = (power + BIAS) if power >= ESUB else 0
    bits = (sign << 63) | (ef << MB) | (M & ((1 << MB) - 1))
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0], dropped


def scale(x, d):
    """x * 2^d, done on the exponent so the significand is left alone (see pack)."""
    p = parts(x)
    if p["kind"] in ("inf", "nan", "zero"):
        return x, 0
    return pack(p["sign"], p["M"], p["q"] + d)


def bit_at(p, place):
    """The bit x holds at 2^place, or None when that place is outside x's window."""
    if place < p["bot"] or place > p["lead"]:
        return None
    return (p["M"] >> (place - p["q"])) & 1


def exact_decimal(sign, M, q):
    """The exact decimal value of (-1)^sign * M * 2^q.

    Every float64 has one and it is finite: M * 2^-k = M * 5^k / 10^k.
    """
    if M == 0:
        return "-0" if sign else "0"
    if q >= 0:
        return ("-" if sign else "") + str(M << q)
    k = -q
    digits = str(M * 5**k).rjust(k + 1, "0")
    whole, frac = digits[:-k], digits[-k:]
    return ("-" if sign else "") + f"{whole}.{frac}".rstrip("0").rstrip(".")


def significand(p):
    """x's significand as an exact decimal string: in [1, 2) normal, below 1 subnormal."""
    e = p["lead"] if p["kind"] == "normal" else ESUB
    return exact_decimal(0, p["M"], p["q"] - e)


# --------------------------------------------------------------- the line of points

def region_points(e, bits):
    """The representable values in [2^e, 2^(e+1)) at `bits` significand bits.

    Evenly spaced by 2^(e-bits): one region's worth of the quantized reals.
    """
    return [math.ldexp(1.0 + j / (1 << bits), e) for j in range(1 << bits)]


def regions_for(e, below, above):
    """The run of regions to draw around region e, clamped to the drawable range.

    Always contains e, so the value always has a region to sit in.
    """
    low = max(ESUB, min(e - below, EDRAW - below - above))
    return list(range(low, min(EDRAW, low + below + above) + 1))


def line_data(x, regions, bits):
    """The point line for one step: (grid points, their hovertext, the value's marker)."""
    xs, text = [], []
    for e in regions:
        step = math.ldexp(1.0, e - bits)
        for v in region_points(e, bits):
            xs.append(v)
            text.append(f"{v!r}<br>region [2^{e}, 2^{e + 1}), spacing 2^{e - bits} = {step!r}")
    return xs, text


# --------------------------------------------------------------- the bars

# Fixed bar positions, so a slider step only ever changes a bar's width (1 to
# show it, 0 to hide it) and never has to move it. That keeps every step's
# update to x / y / hovertext, which both the bars and the points accept.
BIT_BASE = [k - 0.5 for k in range(SB)]
TRACE_ORDER = ("points", "value", "track", "bit_1", "bit_0", "lead")


def bar_data(x):
    """Bar widths and hovertext for the three bit traces, indexed from the leading 1."""
    p = parts(x)
    out = {name: dict(x=[0.0] * SB, hovertext=[""] * SB) for name in ("bit_1", "bit_0", "lead")}
    for k in range(SB):
        place = p["lead"] - k
        bit = bit_at(p, place)
        if bit is None:                     # a subnormal reaches fewer than 53 places
            continue
        which = ("the hidden leading 1, not stored" if k == 0 and p["kind"] == "normal"
                 else f"stored significand bit {k}")
        name = "lead" if k == 0 and bit else f"bit_{bit}"
        out[name]["x"][k] = 1.0
        out[name]["hovertext"][k] = f"bit {k} at 2^{place} = {bit}<br>{which}"
    return out


# --------------------------------------------------------------- one step

def step_data(x, regions, bits):
    """The six traces' x / y / hovertext for one slider step, in TRACE_ORDER."""
    p = parts(x)
    xs, text = line_data(x, regions, bits)
    bars = bar_data(x)
    ulp = p["bot"]
    rows = {
        "points": dict(x=xs, y=[0] * len(xs), hovertext=text),
        "value":  dict(x=[abs(x)], y=[0],
                       hovertext=[f"x = {x!r}<br>region [2^{p['lead']}, 2^{p['lead'] + 1})"
                                  f"<br>real spacing here is the ulp 2^{ulp}"]),
        "track":  dict(x=[float(SB)], y=[0], hovertext=[f"the 53 significand bits of {x!r}"]),
        "bit_1":  dict(x=bars["bit_1"]["x"], y=[0] * SB, hovertext=bars["bit_1"]["hovertext"]),
        "bit_0":  dict(x=bars["bit_0"]["x"], y=[0] * SB, hovertext=bars["bit_0"]["hovertext"]),
        "lead":   dict(x=bars["lead"]["x"], y=[0] * SB, hovertext=bars["lead"]["hovertext"]),
    }
    keys = ("x", "y", "hovertext")
    data = {key: [rows[name][key] for name in TRACE_ORDER] for key in keys}
    return data, list(range(len(TRACE_ORDER)))


def step_layout(x, regions, bits):
    """Layout for one slider step, as plotly's dotted paths: regions, labels, title."""
    p = parts(x)
    lo = math.ldexp(1.0, regions[0])
    hi = math.ldexp(1.0, regions[-1] + 1)

    shapes, annotations = [], []
    for n, e in enumerate(regions):
        left, right = math.ldexp(1.0, e), math.ldexp(1.0, e + 1)
        if n % 2 == 0:                       # shade alternate regions so they read as regions
            shapes.append(dict(type="rect", xref="x", yref="y", layer="below",
                               x0=left, x1=right, y0=-1, y1=1,
                               fillcolor=REGION_FILL, line=dict(width=0)))
        shapes.append(dict(type="line", xref="x", yref="y", x0=left, x1=left,
                           y0=-1, y1=1, line=dict(color=EDGE_COLOR, width=1)))
        annotations.append(dict(
            text=f"2<sup>{e}</sup>", x=left, y=-1, xref="x", yref="y",
            xanchor="center", yanchor="top", yshift=-4, showarrow=False,
            font=dict(family="monospace", size=10, color="#555"),
        ))
        annotations.append(dict(
            text=f"<b>e = {e}</b><br>spacing 2<sup>{e - bits}</sup>",
            x=(left + right) / 2, y=1, xref="x", yref="y",
            xanchor="center", yanchor="bottom", yshift=2, showarrow=False,
            font=dict(family="monospace", size=9,
                      color=VALUE_COLOR if e == p["lead"] else "#8a929b"),
        ))
    shapes.append(dict(type="line", xref="x", yref="y", x0=hi, x1=hi, y0=-1, y1=1,
                       line=dict(color=EDGE_COLOR, width=1)))
    annotations.append(dict(
        text=f"2<sup>{regions[-1] + 1}</sup>", x=hi, y=-1, xref="x", yref="y",
        xanchor="center", yanchor="top", yshift=-4, showarrow=False,
        font=dict(family="monospace", size=10, color="#555"),
    ))
    # the value, named on the line itself
    annotations.append(dict(
        text=f"<b>{x!r}</b>", x=abs(x), y=0, xref="x", yref="y",
        xanchor="center", yanchor="bottom", yshift=14, showarrow=False,
        font=dict(family="monospace", size=12, color=VALUE_COLOR),
    ))
    # the sign, where the eye already is: just left of the leading bit
    annotations.append(dict(
        text="−" if p["sign"] else "+", x=-0.5, y=0, xref="x2", yref="y2",
        xanchor="right", xshift=-3, showarrow=False,
        font=dict(family="monospace", size=14, color="#333"),
    ))
    annotations += panel_titles(bits)

    sign = "−" if p["sign"] else ""
    e_shown = p["lead"] if p["kind"] == "normal" else ESUB
    edges = [math.ldexp(1.0, e) for e in regions] + [hi]
    ticks = [k for k in range(SB) if k % 4 == 0]
    return {
        "xaxis.range": [lo, hi],
        "xaxis.tickvals": edges,
        "xaxis.ticktext": [f"{v:g}" for v in edges],
        "xaxis2.tickvals": ticks,
        "xaxis2.ticktext": [str(k) for k in ticks],
        "shapes": shapes,
        "annotations": annotations,
        "title.text": (
            f"x = {sign}{significand(p)} × 2<sup>{e_shown}</sup> = {x!r}"
            f"<br><span style='font-size:12px'>region [2<sup>{p['lead']}</sup>,"
            f" 2<sup>{p['lead'] + 1}</sup>) = [{math.ldexp(1.0, p['lead'])!r},"
            f" {math.ldexp(1.0, p['lead'] + 1)!r}) · real ulp 2<sup>{p['bot']}</sup>"
            f" · exact value {short_exact(p)}</span>"
        ),
    }


def short_exact(p, limit=46):
    """x's exact decimal, trimmed: 0.1 needs 55 digits and the title has no room."""
    text = exact_decimal(p["sign"], p["M"], p["q"])
    return text if len(text) <= limit else text[:limit] + f"… ({len(text)} chars)"


def panel_titles(bits):
    """The two panel headers. They live in `annotations`, which every step replaces."""
    return [
        dict(text=f"<b>FLOATING-POINT SPACE</b>  ·  the quantized reals, drawn at 1+{bits}"
                  f" significand bits so the points are visible (binary64 packs 2<sup>52</sup> per region)",
             x=0, y=1.0, xref="x domain", yref="paper", xanchor="left", yanchor="bottom",
             yshift=26, showarrow=False, font=dict(family="monospace", size=10, color="#555")),
        dict(text="<b>THE BITS</b>  ·  all 53, from the leading 1",
             x=0, y=1.0, xref="x2 domain", yref="paper", xanchor="left", yanchor="bottom",
             yshift=26, showarrow=False, font=dict(family="monospace", size=10, color="#555")),
    ]


# --------------------------------------------------------------- the figure

BAR_LEGEND = (
    ("leading 1 (2^e, not stored)", LEAD_COLOR, None),
    ("significand bit = 1", BIT_COLOR, None),
    ("significand bit = 0", "white", dict(HATCH, fgcolor=BIT_COLOR)),
)


def base_figure(data):
    """The six traces and the axes. Step-dependent layout is step_layout's job."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.07)
    x, y, text = data["x"], data["y"], data["hovertext"]

    # left: the line of points, and the value sitting on it
    fig.add_trace(go.Scatter(
        x=x[0], y=y[0], hovertext=text[0], mode="markers", hoverinfo="text",
        marker=dict(size=6, color=POINT_COLOR, symbol="line-ns-open",
                    line=dict(color=POINT_COLOR, width=1.4)),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x[1], y=y[1], hovertext=text[1], mode="markers", hoverinfo="text",
        marker=dict(size=15, color=VALUE_COLOR, line=dict(color="white", width=2)),
        showlegend=False,
    ), row=1, col=1)

    # right: the bits
    fig.add_trace(go.Bar(
        x=x[2], y=y[2], base=[-0.5], hovertext=text[2], orientation="h", width=0.5,
        hoverinfo="text", marker=dict(color=TRACK_COLOR), showlegend=False,
    ), row=1, col=2)
    for n, name in enumerate(("bit_1", "bit_0", "lead"), start=3):
        marker = {
            "bit_1": dict(color=BIT_COLOR, line=dict(color="white", width=1)),
            "bit_0": dict(color="white", pattern=dict(HATCH, fgcolor=BIT_COLOR),
                          line=dict(color="white", width=1)),
            "lead": dict(color=LEAD_COLOR, line=dict(color="white", width=1)),
        }[name]
        fig.add_trace(go.Bar(
            x=x[n], y=y[n], base=BIT_BASE, hovertext=text[n], orientation="h", width=0.5,
            hoverinfo="text", marker=marker, showlegend=False,
        ), row=1, col=2)

    # legend proxies, after the data traces so the step indices above stay put
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="representable value",
        marker=dict(size=9, color=POINT_COLOR, symbol="line-ns-open",
                    line=dict(color=POINT_COLOR, width=1.6)),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="x",
        marker=dict(size=12, color=VALUE_COLOR, line=dict(color="white", width=2)),
    ), row=1, col=1)
    for name, color, pattern in BAR_LEGEND:
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name=name,
                             marker=dict(color=color, pattern=pattern,
                                         line=dict(color="#999", width=1))), row=1, col=2)

    fig.update_xaxes(title="value", showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title="significand bit (0 = the leading 1)", tickmode="array",
                     tickfont=dict(size=10), showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        plot_bgcolor="white", barmode="overlay", bargap=0,
        margin=dict(l=60, r=30, t=150, b=120),
        height=430,
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=14)),
        legend=dict(orientation="h", y=-0.42, x=0, font=dict(size=11)),
        hovermode="closest",
    )
    return fig


def exponent_slider(steps, start):
    """A slider over the exponent: each step is x * 2^d, same bits, a different region."""
    return dict(
        active=start, currentvalue=dict(prefix="exponent e = "), pad=dict(t=55),
        steps=[dict(label=str(e), method="update", args=[data, layout, indices])
               for e, data, layout, indices in steps],
    )


def case_views(x0, span, bits, below, above):
    """Start data, start layout and exponent slider for one value."""
    p0 = parts(x0)
    if p0["kind"] != "normal":
        raise ValueError(f"{x0!r} is {p0['kind']}: the point line is drawn over the normal range")

    if not ESUB <= p0["lead"] <= EDRAW:
        raise ValueError(f"{x0!r} sits in region 2^{p0['lead']}, outside the drawable "
                         f"range 2^{ESUB} .. 2^{EDRAW}")
    exponents = [e for e in range(p0["lead"] - span, p0["lead"] + span + 1)
                 if ESUB <= e <= EDRAW]
    steps, start_data, start_layout = [], None, None
    for e in exponents:
        x, _ = scale(x0, e - p0["lead"])
        regions = regions_for(e, below, above)
        data, indices = step_data(x, regions, bits)
        layout = step_layout(x, regions, bits)
        steps.append((e, data, layout, indices))
        if e == p0["lead"]:
            start_data, start_layout = data, layout
    return start_data, start_layout, exponent_slider(steps, exponents.index(p0["lead"]))


def plot_space(values=VALUES, span=10, bits=4, below=1, above=1):
    """One float64: its place among the quantized reals, and its bits.

    `values` maps a label to a finite normal float. `span` is how many regions
    either side of the start the slider reaches. `bits` is the display precision
    of the point line. `below` / `above` are how many neighbouring regions to
    draw, which is what makes the spacing double visibly across the view.

    Plotly controls can't read each other's state, so each dropdown entry swaps
    in its own slider along with its data -- picking a new value resets the
    exponent, the same way the case dropdown in plot_split resets M.
    """
    views = {label: case_views(x, span, bits, below, above) for label, x in values.items()}

    data, layout, slider = next(iter(views.values()))
    fig = base_figure(data)
    # update_layout takes "xaxis_range" for xaxis.range, so swap the dots
    fig.update_layout(**{key.replace(".", "_"): value for key, value in layout.items()})
    fig.update_layout(sliders=[slider])

    if len(views) > 1:
        buttons = []
        for label, (data, layout, slider) in views.items():
            indices = list(range(len(TRACE_ORDER)))
            buttons.append(dict(label=label, method="update",
                                args=[data, {**layout, "sliders": [slider]}, indices]))
        fig.update_layout(updatemenus=[dict(
            buttons=buttons, direction="down",
            x=1, xanchor="right", y=1.02, yanchor="bottom", showactive=True,
        )])
    return fig


if __name__ == "__main__":
    x = 100.0
    p = parts(x)
    print(f"{x!r} = {significand(p)} × 2^{p['lead']} = {p['M']} × 2^{p['q']}")
    print(f"  region [2^{p['lead']}, 2^{p['lead'] + 1}), ulp 2^{p['bot']}")
    print(f"  exact  {exact_decimal(p['sign'], p['M'], p['q'])}")
    for e in (p["lead"] - 1, p["lead"], p["lead"] + 1):
        pts = region_points(e, 4)
        print(f"  region e={e:<3} spacing 2^{e - 4} -> {pts[0]!r}, {pts[1]!r}, ... {pts[-1]!r}")

    plot_space().write_html("space.html")
