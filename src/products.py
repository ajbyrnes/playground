from fractions import Fraction

import math

import numpy as np

from splitting import extract_scalar, extract_vector


def significant_bits(v):
    """How many bits v needs, from its leading 1 to its last 1 (0 for v = 0).

    Every double, and every product of doubles, is an exact fraction with a
    power-of-two denominator. Fraction keeps it in lowest terms, so the
    numerator holds exactly the significant bits, give or take trailing zeros.
    """
    n = abs(Fraction(v).numerator)
    if n == 0:
        return 0
    trailing_zeros = (n & -n).bit_length() - 1
    return (n >> trailing_zeros).bit_length()


def exact_bits(v):
    """The significant bits of an exact value as (power_of_two, bit) pairs, leading bit first.

    Like floatbits.significand_bits, but for any value with a power-of-two
    denominator (e.g. an exact product of doubles), however many bits it has.
    Stops at the last 1 bit. Returns [] for 0.
    """
    v = Fraction(v)
    n = abs(v.numerator)
    if n == 0:
        return []
    k = v.denominator.bit_length() - 1  # v = ±n / 2^k
    top = n.bit_length() - 1 - k        # power of two of the leading bit
    count = significant_bits(v)
    return [(top - i, (n >> (n.bit_length() - 1 - i)) & 1) for i in range(count)]


def padded_bits(v):
    """exact_bits, padded with trailing 0 bits to at least 53, the width of a double.

    That way an exact value that fits in a double is drawn just like one, and
    only values needing more than 53 bits look longer.
    """
    bits = exact_bits(v)
    if bits and len(bits) < 53:
        last = bits[-1][0]
        bits += [(last - i, 0) for i in range(1, 53 - len(bits) + 1)]
    return bits


def product_info(a, b):
    """a * b in floating point vs exactly: the exact product's bit count and the rounding error."""
    exact = Fraction(a) * Fraction(b)
    rounded = a * b
    return dict(
        a=a, b=b, rounded=rounded,
        exact_bits=significant_bits(exact),
        error=Fraction(rounded) - exact,  # 0 exactly when the product fit in 53 bits
    )


def split_products(x, y, M):
    """Split x and y with extract_scalar, then multiply every pair of pieces."""
    x_hi, x_lo = (float(v) for v in extract_scalar(x, M))
    y_hi, y_lo = (float(v) for v in extract_scalar(y, M))
    pieces = {
        "x_hi·y_hi": (x_hi, y_hi),
        "x_hi·y_lo": (x_hi, y_lo),
        "x_lo·y_hi": (x_lo, y_hi),
        "x_lo·y_lo": (x_lo, y_lo),
    }
    return {name: product_info(a, b) for name, (a, b) in pieces.items()}


def fl_dot(xs, ys):
    """Dot product in plain floating point, in order (no FMA, no reordering, unlike BLAS)."""
    total = 0.0
    for a, b in zip(xs, ys):
        total += float(a) * float(b)  # rounds twice per term: the product, then the running sum
    return total


def split_M(n):
    """M that makes x_hi^T y_hi exact for length-n vectors (paper's equation (6) and (8)).

    x_hi keeps alpha = floor((53 - ceil(log2 n)) / 2) bits, so each product
    needs at most 2*alpha bits and their sum at most 2*alpha + log2 n <= 53.
    x_hi keeps 53 - M bits, so M = 53 - alpha.
    """
    alpha = (53 - math.ceil(math.log2(n))) // 2
    return 53 - alpha


def split_dot(xs, ys, M=None):
    """Equation (7): x^T y = x_hi^T y_hi + x_hi^T y_lo + x_lo^T y, each part in plain floating point.

    Returns the three computed parts, their sum, whether the first part came out
    exact, and the split pieces themselves.
    """
    M = M or split_M(len(xs))
    x_hi, x_lo = extract_vector(np.asarray(xs, dtype=float), M)
    y_hi, y_lo = extract_vector(np.asarray(ys, dtype=float), M)

    parts = [fl_dot(x_hi, y_hi), fl_dot(x_hi, y_lo), fl_dot(x_lo, ys)]
    hi_exact = Fraction(parts[0]) == sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x_hi, y_hi))
    return dict(
        M=M, parts=parts, total=parts[0] + parts[1] + parts[2], hi_exact=hi_exact,
        x_hi=x_hi, x_lo=x_lo, y_hi=y_hi, y_lo=y_lo,
    )


def dot_info(xs, ys):
    """A dot product in plain floating point vs exactly.

    The condition number sum|x_i y_i| / |sum x_i y_i| says how much cancellation
    amplifies rounding errors: plain floating point is off by roughly
    (condition number) × 1.1e-16, relative to the answer.
    """
    rounded = fl_dot(xs, ys)
    terms = [Fraction(a) * Fraction(b) for a, b in zip(xs, ys)]
    exact = sum(terms)
    return dict(
        rounded=rounded,
        exact=exact,
        error=Fraction(rounded) - exact,
        condition=sum(abs(t) for t in terms) / abs(exact) if exact else float("inf"),
    )


DOT_CASES = {
    "x·3 − 1 with x = fl(1/3)": ([1 / 3, -1.0], [3.0, 1.0]),
    "1e16 + 1 − 1e16": ([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]),
    "0.1·0.1 + 0.2·0.2 − 0.05": ([0.1, 0.2, -0.05], [0.1, 0.2, 1.0]),
}


if __name__ == "__main__":
    for name, (xs, ys) in DOT_CASES.items():
        d = dot_info(xs, ys)
        rel = abs(d["error"] / d["exact"]) if d["exact"] else float("nan")
        print(f"{name}: plain = {d['rounded']!r}, exact ≈ {float(d['exact']):.17g}, "
              f"relative error = {float(rel):.2g}, condition ≈ {float(d['condition']):.2g}")
    print()

    x, y = 0.1, 1 / 3

    plain = product_info(x, y)
    print(f"x = {x!r} ({significant_bits(x)} bits), y = {y!r} ({significant_bits(y)} bits)")
    print(f"x·y exact needs {plain['exact_bits']} bits; rounded to 53, error = {float(plain['error']):.3g}\n")

    for M in (20, 27):
        print(f"M = {M}")
        parts = split_products(x, y, M)
        for name, p in parts.items():
            print(f"  {name}: {significant_bits(p['a']):>2} × {significant_bits(p['b']):>2} bits "
                  f"-> exact needs {p['exact_bits']:>3} bits, error = {float(p['error']):.3g}")
        kept_exactly = sum(Fraction(p["rounded"]) for p in parts.values())
        print(f"  the four rounded products, summed exactly, equal x·y exactly: "
              f"{kept_exactly == Fraction(x) * Fraction(y)}\n")
