"""Sigma as a lens: it sets a resolution, and M is the zoom.

Sigma never touches x. It fixes how finely x can be seen. fl(sigma + x) has to
land on a value representable in sigma's region, where consecutive values sit
delta = 2^(e_sigma - 52) apart, so sigma behaves like a lens that cannot tell
apart two positions closer together than delta. Raising M grows sigma, which
widens delta: you zoom out, and more of x's own neighbouring values crowd into
one indistinguishable cell. Lowering M keeps you zoomed in, and at small enough
M nothing is lost at all.

Left: one line. x's own representable neighbours are ticked along it, and the
lens's delta-wide cells are drawn over them. x's cell is lit. Everything inside
that cell looks identical to the lens, and the only position it can report is
the cell's centre -- which is x_hi. x's offset from that centre is x_lo, drawn
inside the cell. So "the low bits fall away" is not an operation performed on
x: those bits describe detail finer than one cell, and the lens has no way to
express them.

Right: x's 53 bits with the cut at delta, blue for the bits above it and orange
for the bits below, the same colours and the same cut as plot_split. The cut and
the cell width are one number seen two ways.

A caveat the picture states rather than hides: round-to-nearest can round away
from zero, and then x_lo is not simply the low bits but those bits minus one
whole cell. x = x_hi + x_lo stays exact either way.

Every tick drawn is a real representable value. A cell can hold up to 2^52 of
x's own positions, more than any screen has pixels, so x's lattice is decimated
to fit: the row draws every 2^stride-th of its points and says so. A decimated
uniform lattice is still a uniform lattice of real floats, so nothing is
invented -- there are only more ticks than shown.
"""

import math
import struct
from fractions import Fraction

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from floatbits import COLORS, HATCH, TRACK_COLOR
from plot_space import MB, SB, bit_at, exact_decimal, parts, significand

HI_COLOR, LO_COLOR = COLORS["blue"], COLORS["orange"]
VALUE_COLOR = COLORS["red"]
SIGMA_COLOR = COLORS["aqua"]
POINT_COLOR = "#b9c0c9"
CUT_COLOR = "#222"



# Panel 1 lives on a place-value axis, where a float is a 53-place window: its
# exponent anchors the top and its resolution falls out 53 places below. That
# axis is logarithmic in value, so the whole M range fits on it at once.
# Panel 2 lives in value space, where only a few cells fit at a time.
Y_SIGMA, Y_X = 1.0, 0.0          # the two windows in panel 1
WIN_H = 0.3                      # half-height of a window bar

# Panel 2 holds three cells of fixed width with x_hi pinned at the centre, and
# its two neighbouring floats either side. The axis is in units of a cell, so it
# never moves. What changes with M is the real values printed on the cells, and
# where x sits inside its own cell -- which it does differently at every M.
CELL_H = 0.60                    # half-height of a cell
Y_DELTA = 0.95                   # the bar that measures one cell's width
Y_XLO, Y_ULP = 0.28, -0.32       # x_lo and ulp(x), both drawn inside x's own cell
MAX_SUBDIV = 5                   # draw x's own values inside the cell up to 2^this
MIN_SLIVER = 0.004               # narrower than this and ulp(x) is not drawable
PANEL_Y = (-1.12, 1.20)

# A cell is bounded left and right and open top and bottom: its edges are the
# two values where the rounding flips to a neighbouring float, and nothing
# bounds it vertically, so only the edges are drawn.
CELL_NEIGHBOUR = "rgba(120,132,145,0.06)"  # the floats either side of x_hi
CELL_LIT = "rgba(42,120,214,0.10)"         # the cell x falls in
EDGE_FAINT = "#ccd2d9"                     # a neighbour's outer edge

VALUES = {
    "pi": math.pi,
    "0.1": 0.1,
    "1 + 2^-52": 1.0 + 2.0**-52,
    "-pi": -math.pi,
}


# --------------------------------------------------------------- the split

def sigma_for(x, M):
    """sigma from equation (1), computed the same way extract_scalar does."""
    return 2.0**M * 2.0 ** math.ceil(math.log2(abs(x)))


def coarse_place(x, M):
    """The place value of sigma's lattice: the lowest bit fl(sigma + x) can hold.

    sigma + x lands in sigma's own region when x > 0. When x < 0 it falls one
    region lower, where the spacing is one power of two finer, which is why a
    negative element keeps one extra bit (see plot_vector.element_cut).
    """
    e_sigma = M + math.ceil(math.log2(abs(x)))
    return e_sigma - 52 - (1 if x < 0 else 0)


def snap(x, place):
    """x rounded to the nearest multiple of 2^place, ties to even: what fl does.

    Done on exact integers so the drawing and the arithmetic cannot disagree.
    """
    r = Fraction(x) / Fraction(2) ** place
    whole = r.numerator // r.denominator
    rest = r - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2):
        whole += 1
    return float(Fraction(whole) * Fraction(2) ** place)


def split(x, M):
    """(x_hi, x_lo, place): x snapped to sigma's lattice, the residual, and the cut."""
    place = coarse_place(x, M)
    x_hi = snap(x, place)
    return x_hi, x - x_hi, place


def cell_edges(x, M):
    """The cell x falls in: every real in it is pinned to the same float, x_hi.

    Returns (low, high, x_hi, delta) as exact Fractions. These are the rounding
    boundaries of the destination window's resolution, so the cell is literally
    "the set of reals that fl(sigma + .) - sigma maps to x_hi".
    """
    place = coarse_place(x, M)
    delta = Fraction(2) ** place
    centre = Fraction(snap(x, place))
    return centre - delta / 2, centre + delta / 2, centre, delta


def as_text(value, limit=26):
    """An exact Fraction as a decimal string, preferring a float's own repr.

    A cell edge is x_hi +/- delta/2 and is normally exactly representable, so
    repr gives the short round-tripping form. When it is not (delta/2 can fall
    below x's own ulp), fall back to the exact decimal rather than print a value
    that is quietly rounded.
    """
    near = float(value)
    if Fraction(near) == value:
        return repr(near)
    p = parts(float(value))
    text = exact_decimal(0 if value >= 0 else 1, p["M"], p["q"])
    return text if len(text) <= limit else text[:limit] + "…"


def bits_each_side(p, place):
    """How many of x's 53 bits sit at or above the cut (x_hi's side) and below it."""
    n_hi = min(max(p["lead"] - place + 1, 0), SB)
    return n_hi, SB - n_hi


# --------------------------------------------------------------- the two windows

def sigma_exp(x, M):
    """The exponent of sigma = 2^M * 2^ceil(log2|x|)."""
    return M + math.ceil(math.log2(abs(x)))


def dest_window(x, M):
    """The 53-place window fl(sigma + x) lives in, as (bottom, top).

    This is the whole mechanism in one line. The window's exponent anchors its
    top, and its resolution is 53 places below that -- so a window parked high
    up has a coarse floor. x can only keep the bits of its own window that fall
    inside this one; the rest are below the floor and have nowhere to go.

    For x < 0 the sum lands one region lower than sigma, so the window sits one
    place down and keeps one extra bit (see plot_vector.element_cut).
    """
    place = coarse_place(x, M)
    return place, place + MB


def nesting(x, M):
    """How many places the destination window's floor sits above x's own floor.

    This single number is the whole relationship: it is how many of x's own
    places get stranded, and 2^nesting is how much coarser the destination's
    resolution is than x's.
    """
    return coarse_place(x, M) - parts(x)["bot"]


def place_axis(x, Ms):
    """Panel 1's fixed axis, measured in places below x's leading bit.

    Measuring from x's leading bit rather than in absolute powers of two keeps
    x's own window parked at 0..52 for every value and every M, so the bars never
    have to move -- only the destination window, drawn as a shape, slides. The
    tick labels still read as real powers of two.
    """
    return -(max(nesting(x, M) for M in Ms) + 2), SB + 1


# --------------------------------------------------------------- one step

# Panel 1 is measured in places below x's leading bit, so x's own window is
# always 0..52 and these bases are constant. A step only changes a bar's width
# (1 to show it, 0 to hide it). The destination window moves, so it is a shape.
# Panel 2 is in units of one cell, so its two points carry everything that moves.
TRACE_ORDER = ("x_track", "hi_1", "hi_0", "lo_1", "lo_0",
               "x_in_cell", "x_hi_centre", "neighbours")
BIT_BASE = [k - 0.5 for k in range(SB)]


def bar_data(p, place):
    """Bar widths and hovertext for x's 53 places, split at the destination's floor."""
    out = {name: dict(x=[0.0] * SB, hovertext=[""] * SB)
           for name in ("hi_1", "hi_0", "lo_1", "lo_0")}
    for k in range(SB):
        at = p["lead"] - k
        bit = bit_at(p, at)
        if bit is None:
            continue
        side = "hi" if at >= place else "lo"
        out[f"{side}_{bit}"]["x"][k] = 1.0
        out[f"{side}_{bit}"]["hovertext"][k] = (
            f"2^{at} · place {k} = {bit} · "
            + ("kept in x_hi" if side == "hi" else "stranded in x_lo"))
    return out


def step_data(x, M):
    """The seven traces' x / y / hovertext for one M, in TRACE_ORDER."""
    p = parts(x)
    x_hi, x_lo, place = split(x, M)
    low, high, centre, delta = cell_edges(x, M)
    nest = nesting(x, M)
    frac = float(Fraction(x) - centre) / float(delta)   # x's place in the cell, in cells

    bars = bar_data(p, place)
    rows = {
        "x_track": dict(x=[float(SB)], y=[Y_X],
                        hovertext=[f"x's own window: 2^{p['lead']} down to 2^{p['bot']},"
                                   f" 53 places"]),
        "x_in_cell": dict(x=[frac], y=[0],
                          hovertext=[f"x = {x!r}<br>{frac:+.4f} of a cell from the centre"
                                     f"<br>2^{nest} of x's own values share this cell"]),
        "x_hi_centre": dict(x=[0.0], y=[0],
                            hovertext=[f"x_hi = {x_hi!r}<br>the one float every real in"
                                       f" this cell is pinned to"]),
        "neighbours": dict(x=[-1.0, 1.0], y=[0, 0],
                           hovertext=[f"{as_text(centre - delta)}<br>the next float below"
                                      f" x_hi in this window",
                                      f"{as_text(centre + delta)}<br>the next float above"
                                      f" x_hi in this window"]),
    }
    for name in ("hi_1", "hi_0", "lo_1", "lo_0"):
        rows[name] = dict(x=bars[name]["x"], y=[Y_X] * SB,
                          hovertext=bars[name]["hovertext"])

    keys = ("x", "y", "hovertext")
    data = {key: [rows[name][key] for name in TRACE_ORDER] for key in keys}
    extra = dict(x_hi=x_hi, x_lo=x_lo, place=place, win=dest_window(x, M), nest=nest,
                 low=low, high=high, centre=centre, delta=delta, frac=frac)
    return data, list(range(len(TRACE_ORDER))), extra


def step_layout(x, M, e, axis):
    """Layout for one M: the window sliding in panel 1, the cell's edges in panel 2.

    Neither axis ever moves. In panel 1 the destination window slides, lifting
    its floor through x's window and stranding the places underneath. In panel 2
    the cell is always the full width of the panel, so what changes is the pair
    of real values on its edges -- the rounding boundaries -- and how much of x's
    own detail fits between them.
    """
    p = parts(x)
    sigma = sigma_for(x, M)
    n_hi, n_lo = bits_each_side(p, e["place"])
    bot, top = e["win"]
    nest, frac = e["nest"], e["frac"]
    u_top, u_floor = -nest, SB - 1 - nest
    a_lo, a_hi = axis

    def mono(size, color, bold=False):
        return dict(family="monospace", size=size, color=color)

    def caption(ax, text, size, color, yshift, top_edge=True):
        return dict(text=text, xref=f"{ax} domain", x=0,
                    yref=f"{ax.replace('x', 'y')} domain", y=1 if top_edge else 0,
                    xanchor="left", yanchor="bottom" if top_edge else "top",
                    yshift=yshift, showarrow=False, font=mono(size, color))

    shapes, ann = [], []

    # ---- panel 1: the destination window, parked where M puts it ----
    shapes += [
        dict(type="rect", xref="x", yref="y", layer="below",
             x0=u_top - 0.5, x1=u_floor + 0.5, y0=Y_SIGMA - WIN_H, y1=Y_SIGMA + WIN_H,
             fillcolor="rgba(27,175,122,0.13)", line=dict(color=SIGMA_COLOR, width=1.2)),
        dict(type="rect", xref="x", yref="y",                      # sigma's leading 1
             x0=u_top - 0.5, x1=u_top + 0.5, y0=Y_SIGMA - WIN_H, y1=Y_SIGMA + WIN_H,
             fillcolor=SIGMA_COLOR, line=dict(width=0)),
        dict(type="line", xref="x", yref="y",                      # its floor = the cut
             x0=u_floor + 0.5, x1=u_floor + 0.5, y0=Y_X - 0.6, y1=Y_SIGMA + WIN_H,
             line=dict(color=CUT_COLOR, width=2, dash="dot")),
    ]
    if n_hi:
        ann.append(dict(text=f"{n_hi} of x's places fall inside the window → x_hi",
                        x=max(u_floor, 0) / 2, y=Y_X + WIN_H, xref="x", yref="y",
                        xanchor="center", yanchor="bottom", yshift=2, showarrow=False,
                        font=mono(9, HI_COLOR)))
    else:
        ann.append(dict(text="the window is parked entirely above x — it can hold none"
                             " of x's places",
                        x=SB / 2, y=Y_X + WIN_H, xref="x", yref="y", xanchor="center",
                        yanchor="bottom", yshift=2, showarrow=False, font=mono(9, LO_COLOR)))
    if n_lo:
        ann.append(dict(text=f"{n_lo} fall below its floor → x_lo",
                        x=(max(u_floor + 1, 0) + SB) / 2, y=Y_X - WIN_H, xref="x", yref="y",
                        xanchor="center", yanchor="top", yshift=-2, showarrow=False,
                        font=mono(9, LO_COLOR)))
    ann += [
        dict(text=f"σ + x lands here · exponent 2<sup>{top}</sup> anchors the top,"
                  f" so 53 places down its resolution is δ = 2<sup>{bot}</sup>",
             x=u_top - 0.5, y=Y_SIGMA + WIN_H, xref="x", yref="y", xanchor="left",
             yanchor="bottom", yshift=3, showarrow=False, font=mono(9, SIGMA_COLOR)),
        dict(text=f"x · exponent 2<sup>{p['lead']}</sup> anchors the top, so 53 places"
                  f" down its resolution is ulp = 2<sup>{p['bot']}</sup>",
             x=-0.5, y=Y_X - WIN_H, xref="x", yref="y", xanchor="left",
             yanchor="top", yshift=-18, showarrow=False, font=mono(9, "#444")),
        dict(text="−" if p["sign"] else "+", x=-0.5, y=Y_X, xref="x", yref="y",
             xanchor="right", xshift=-2, showarrow=False, font=mono(13, "#333")),
        caption("x", "<b>1 · TWO 53-PLACE WINDOWS</b> · every float is one of these:"
                     " the exponent anchors the top, the resolution follows 53 places down",
                10, "#444", 24),
        caption("x", f"M = {M} parks the destination 2<sup>{nest}</sup>× higher than x,"
                     f" so its floor sits {nest} places above x's", 9, "#6b7480", 10),
    ]
    ticks = [k for k in range(a_lo, a_hi + 1) if (p["lead"] - k) % 4 == 0]

    # ---- panel 2: three cells of fixed width, x_hi pinned at the centre ----
    for k in (-1, 0, 1):
        mid = k == 0
        shapes.append(dict(                                  # the cell's body
            type="rect", xref="x2", yref="y2", layer="below", x0=k - 0.5, x1=k + 0.5,
            y0=-CELL_H, y1=CELL_H, fillcolor=CELL_LIT if mid else CELL_NEIGHBOUR,
            line=dict(width=0)))
        shapes.append(dict(type="line", xref="x2", yref="y2", x0=k, x1=k,
                           y0=-CELL_H, y1=CELL_H,            # the float it pins to
                           line=dict(color=HI_COLOR if mid else "#aab2bb",
                                     width=1.2 if mid else 0.9, dash="dash")))
    for u in (-1.5, -0.5, 0.5, 1.5):                         # where rounding flips
        inner = abs(u) < 1
        shapes.append(dict(type="line", xref="x2", yref="y2", x0=u, x1=u,
                           y0=-CELL_H, y1=CELL_H,
                           line=dict(color=HI_COLOR if inner else EDGE_FAINT,
                                     width=2 if inner else 1)))

    # x's cell's width, measured
    shapes.append(dict(type="line", xref="x2", yref="y2", x0=-0.5, x1=0.5,
                       y0=Y_DELTA, y1=Y_DELTA, line=dict(color="#8a929b", width=1)))
    for u in (-0.5, 0.5):
        shapes.append(dict(type="line", xref="x2", yref="y2", x0=u, x1=u,
                           y0=Y_DELTA - 0.06, y1=Y_DELTA + 0.06,
                           line=dict(color="#8a929b", width=1)))

    # x's own representable values inside its cell, while there are few enough
    if nest <= MAX_SUBDIV:
        for k in range(1 << nest):
            u = -0.5 + (k + 0.5) / (1 << nest)
            shapes.append(dict(type="line", xref="x2", yref="y2", x0=u, x1=u,
                               y0=-0.19, y1=-0.03, line=dict(color=POINT_COLOR, width=1)))

    # x_lo, to scale: the gap from the cell's centre out to where x really is
    left, right = sorted((0.0, frac))
    if frac != 0.0:
        shapes.append(dict(type="rect", xref="x2", yref="y2", x0=left, x1=right,
                           y0=Y_XLO - 0.075, y1=Y_XLO + 0.075, fillcolor=LO_COLOR,
                           opacity=0.65, line=dict(width=0)))

    # ulp(x): x's own resolution, at this cell's scale
    sliver = 1.0 / (1 << nest) if nest < 40 else 0.0
    if sliver >= MIN_SLIVER:
        shapes.append(dict(type="rect", xref="x2", yref="y2",
                           x0=frac - sliver / 2, x1=frac + sliver / 2,
                           y0=Y_ULP - 0.06, y1=Y_ULP + 0.06,
                           fillcolor=VALUE_COLOR, opacity=0.5, line=dict(width=0)))

    # Every label below gets a row of its own: the edges sit above the cells,
    # the floats below them, and x is labelled off to one side of its own dot.
    big = dict(family="monospace", size=12, color=HI_COLOR)
    small = dict(family="monospace", size=10, color="#6b7480")
    note = dict(family="monospace", size=9, color=LO_COLOR)

    def label(text, u, y, font, above=True, shift=4):
        return dict(text=text, x=u, y=y, xref="x2", yref="y2", xanchor="center",
                    yanchor="bottom" if above else "top",
                    yshift=shift if above else -shift, showarrow=False, font=font)

    ann += [
        caption("x2", "<b>2 · THREE NEIGHBOURING FLOATS, EACH WITH ITS CELL</b> · a cell is"
                      " every real number that gets pinned to the float at its centre",
                10, "#444", 40),
        caption("x2", f"the cells never change width — their edges do. M = {M} makes each"
                      f" one δ = 2<sup>{e['place']}</sup> wide", 9, "#6b7480", 26),
        label(f"cell width δ = {float(e['delta'])!r}", 0.0, Y_DELTA, small),
        # the two edges of x's cell, above it and big, because they are what moves
        label(f"<b>{as_text(e['low'])}</b>", -0.5, CELL_H, big),
        label(f"<b>{as_text(e['high'])}</b>", 0.5, CELL_H, big),
        # the three floats, under the cells they pin
        label(f"<b>{e['x_hi']!r}</b>", 0.0, -CELL_H, big, above=False, shift=6),
        label(as_text(e["centre"] - e["delta"]), -1.0, -CELL_H, small, above=False, shift=6),
        label(as_text(e["centre"] + e["delta"]), 1.0, -CELL_H, small, above=False, shift=6),
        # x itself, set beside its dot on the side away from the centre
        dict(text=f"x = {x!r}", x=frac, y=0, xref="x2", yref="y2", yanchor="middle",
             xanchor="right" if frac <= 0 else "left", xshift=-10 if frac <= 0 else 10,
             showarrow=False, font=dict(family="monospace", size=10, color=VALUE_COLOR)),
    ]
    ann.append(
        label(f"x_lo = {e['x_lo']!r} · {abs(frac):.4f} of a cell", (left + right) / 2,
              Y_XLO + 0.075, note, shift=3) if frac != 0.0 else
        label("x_lo = 0 · x sits exactly on x_hi", 0.0, Y_XLO + 0.075, note, shift=3))
    drawn = sliver >= MIN_SLIVER
    ann.append(label(
        f"ulp(x) = 2<sup>{p['bot']}</sup> · one part in 2<sup>{nest}</sup> of a cell,"
        f" so 2<sup>{nest}</sup> of x's own values share this one"
        + ("" if drawn else ", far too fine to draw"),
        min(max(frac, -0.7), 0.7) if drawn else 0.0, Y_ULP - 0.06,
        dict(family="monospace", size=9, color=VALUE_COLOR), above=False, shift=3))

    return {
        "xaxis.range": [a_lo - 0.5, a_hi + 0.5],
        "xaxis.tickvals": ticks,
        "xaxis.ticktext": [f"2<sup>{p['lead'] - k}</sup>" for k in ticks],
        "xaxis2.range": [-1.58, 1.58],
        "xaxis2.tickvals": [-1, -0.5, 0, 0.5, 1],
        "xaxis2.ticktext": ["x_hi − δ", "−δ/2", "x_hi", "+δ/2", "x_hi + δ"],
        "shapes": shapes,
        "annotations": ann,
        "title.text": (
            f"x = {x!r} = {significand(p)} × 2<sup>{p['lead']}</sup>"
            f"<br><span style='font-size:12px'>M = {M} → σ = {sigma!r} → σ + x lands in the"
            f" window 2<sup>{top}</sup> … 2<sup>{bot}</sup>, whose resolution"
            f" δ = 2<sup>{bot}</sup> is 2<sup>{nest}</sup>× coarser than x's"
            f" ulp 2<sup>{p['bot']}</sup></span>"
            f"<br><span style='font-size:12px'>x_hi = {e['x_hi']!r} ({n_hi} places kept)"
            f" · x_lo = {e['x_lo']!r} ({n_lo} places stranded below the floor)"
            f" · x = x_hi + x_lo exactly</span>"
        ),
    }


# --------------------------------------------------------------- the figure

def base_figure(data):
    """The eight traces and two panels. Step-dependent layout is step_layout's job."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.22)
    xs, ys, text = data["x"], data["y"], data["hovertext"]

    # panel 1: x's own 53-place window, and its bits
    fig.add_trace(go.Bar(x=xs[0], y=ys[0], base=[-0.5], hovertext=text[0], orientation="h",
                         width=2 * WIN_H, hoverinfo="text", marker=dict(color=TRACK_COLOR),
                         showlegend=False), row=1, col=1)
    fills = {
        "hi_1": dict(color=HI_COLOR, line=dict(color="white", width=1)),
        "hi_0": dict(color="white", pattern=dict(HATCH, fgcolor=HI_COLOR),
                     line=dict(color="white", width=1)),
        "lo_1": dict(color=LO_COLOR, line=dict(color="white", width=1)),
        "lo_0": dict(color="white", pattern=dict(HATCH, fgcolor=LO_COLOR),
                     line=dict(color="white", width=1)),
    }
    for n, name in enumerate(("hi_1", "hi_0", "lo_1", "lo_0"), start=1):
        fig.add_trace(go.Bar(x=xs[n], y=ys[n], base=BIT_BASE, hovertext=text[n],
                             orientation="h", width=2 * WIN_H, hoverinfo="text",
                             marker=fills[name], showlegend=False), row=1, col=1)

    # panel 2: the same floor, in value space
    def dots(n, **marker):
        fig.add_trace(go.Scatter(x=xs[n], y=ys[n], hovertext=text[n], mode="markers",
                                 hoverinfo="text", marker=marker, showlegend=False),
                      row=2, col=1)

    dots(5, size=12, color=VALUE_COLOR, line=dict(color="white", width=1.5))
    dots(6, size=20, color="rgba(0,0,0,0)", symbol="circle-open",
         line=dict(color=HI_COLOR, width=2.5))
    dots(7, size=13, color="rgba(0,0,0,0)", symbol="circle-open",
         line=dict(color="#8a929b", width=1.8))

    # legend proxies, after the data traces so the step indices stay put
    for name, color, pattern in (("place kept → x_hi", HI_COLOR, None),
                                 ("place stranded → x_lo", LO_COLOR, None),
                                 ("0 bit", "white", dict(HATCH, fgcolor="#555"))):
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name=name,
                             marker=dict(color=color, pattern=pattern,
                                         line=dict(color="#999", width=1))), row=1, col=1)
    for name, marker in (
        ("x, where it really is", dict(size=11, color=VALUE_COLOR,
                                       line=dict(color="white", width=1.5))),
        ("x_hi, the float its cell is pinned to", dict(size=15, color="rgba(0,0,0,0)",
                                                       symbol="circle-open",
                                                       line=dict(color=HI_COLOR, width=2.5))),
        ("the neighbouring floats", dict(size=12, color="rgba(0,0,0,0)",
                                         symbol="circle-open",
                                         line=dict(color="#8a929b", width=1.8))),
    ):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=marker), row=2, col=1)

    mono10 = dict(family="monospace", size=10)
    fig.update_xaxes(title="place value · x's window is the right-hand block",
                     tickmode="array", tickangle=0, tickfont=mono10,
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title="value · the cells are fixed; their edges are not",
                     title_standoff=42, tickmode="array", tickfont=mono10,
                     showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(range=[Y_X - 0.95, Y_SIGMA + 0.75], showgrid=False,
                     zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=list(PANEL_Y), showgrid=False, zeroline=False,
                     showticklabels=False, row=2, col=1)
    fig.update_layout(
        plot_bgcolor="white", barmode="overlay", bargap=0,
        margin=dict(l=60, r=30, t=150, b=210), height=900,
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14)),
        legend=dict(orientation="h", y=-0.19, x=0, font=dict(size=10)),
        hovermode="closest",
    )
    return fig


def m_slider(steps, active):
    """The zoom: each step parks the destination window one place higher."""
    return dict(
        active=active, currentvalue=dict(prefix="M = "), pad=dict(t=110),
        steps=[dict(label=str(M), method="update", args=[data, layout, indices])
               for M, data, layout, indices in steps],
    )


def case_views(x, Ms, start_M):
    """Start data, start layout and the M slider for one value."""
    if parts(x)["kind"] != "normal" or x == 0:
        raise ValueError(f"{x!r}: sigma is defined from log2|x|, so x must be normal and nonzero")
    if start_M not in Ms:
        raise ValueError(f"start_M={start_M} is not in Ms")

    axis = place_axis(x, Ms)        # one fixed place axis for the whole slider
    steps, start = [], None
    for M in Ms:
        data, indices, extra = step_data(x, M)
        layout = step_layout(x, M, extra, axis)
        steps.append((M, data, layout, indices))
        if M == start_M:
            start = (data, layout)
    return start[0], start[1], m_slider(steps, list(Ms).index(start_M))


def plot_sigma(values=VALUES, Ms=range(0, 54), start_M=26):
    """Sigma as a lens: M parks a 53-place window, and its floor strands x's low bits.

    `values` maps a label to a normal nonzero float. `Ms` is the slider's range --
    the full 0..53 works, because panel 1 is drawn on a place-value axis, which
    is logarithmic in value and so has room for every M at once. `start_M` is
    where it opens. `zoom_ref` fixes panel 2's field of view (default `start_M`):
    around it the cells visibly grow, and beyond it the view opens up, since a
    screen only holds about six doublings of cell width. `span` sets the cells
    visible at the reference, `limit` how many contiguous cells panel 2 may draw.

    Plotly controls can't read each other's state, so each dropdown entry swaps
    in its own slider along with its data -- picking a new value resets M, the
    same way the case dropdown in plot_split does.
    """
    Ms = list(Ms)
    views = {label: case_views(x, Ms, start_M) for label, x in values.items()}

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
    from splitting import extract_scalar

    # the picture and the algorithm have to agree, or the picture is a lie
    for x in VALUES.values():
        for M in range(0, 54):
            x_hi, x_lo, place = split(x, M)
            want_hi, want_lo = extract_scalar(x, M)
            assert x_hi == float(want_hi) and x_lo == float(want_lo), (x, M)
    print("split() matches extract_scalar for every value at every M: ok")

    x = math.pi
    p = parts(x)
    print(f"\nx = {x!r} · window 2^{p['lead']} … 2^{p['bot']} · ulp 2^{p['bot']}")
    print(f"{'M':>4} {'sigma':>16} {'window top':>11} {'floor δ':>9} "
          f"{'places above x':>15} {'kept':>5} {'stranded':>9}")
    for M in (0, 1, 10, 26, 40, 52, 53):
        bot, top = dest_window(x, M)
        n_hi, n_lo = bits_each_side(p, bot)
        print(f"{M:>4} {sigma_for(x, M)!r:>16} {('2^%d' % top):>11} {('2^%d' % bot):>9} "
              f"{nesting(x, M):>15} {n_hi:>5} {n_lo:>9}")
    print("\nThe exponent anchors each window's top; the resolution follows 53 places down.")

    plot_sigma().write_html("sigma.html")
