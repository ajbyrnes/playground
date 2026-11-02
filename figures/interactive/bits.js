// Shared by extract_scalar.html and extract_vector.html: reading a double's
// bits, formatting values, and drawing bit rows split by a cut.

// The 53 significand bits of x as [power_of_two, bit] pairs, leading bit
// first, like floatbits.py's significand_bits. [] for 0.
function significandBits(x) {
  if (x === 0) return [];

  // the raw 64 bits, as a BigInt: JS's | & >> only work on 32 bits
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const bits = view.getBigUint64(0);

  const exponent = Number((bits >> 52n) & 0x7FFn);
  const mantissa = bits & ((1n << 52n) - 1n);
  const top = exponent - 1023;                 // power of two of the leading bit
  const significand = (1n << 52n) | mantissa;  // put back the hidden leading 1

  return Array.from({ length: 53 }, (_, i) =>
    [top - i, Number((significand >> BigInt(52 - i)) & 1n)]);
}

// Power of two of the lowest bit fl(σ + x) can hold: x's bits below this
// are rounded away, so x_hi keeps the bits at or above it.
function cutPower(s) {
  return significandBits(s)[0][0] - 52;
}

// Fixed-width scientific notation, so the digits don't shift as values change:
// sign column (space for +), 17 significant digits, then the power of ten.
function formatValue(v) {
  const [mantissa, exp] = Math.abs(v).toExponential(16).split("e");
  return `${v < 0 ? "−" : " "}${mantissa} × 10<sup>${Number(exp)}</sup>`;
}

// x as ±m × 2^E with m in [1, 2). E is the leading bit's power; dividing by
// 2^E is exact (only the exponent changes).
function formatSignificand(x) {
  if (x === 0) return " 0";
  const E = significandBits(x)[0][0];
  const m = Math.abs(x) / 2 ** E;
  return `${x < 0 ? "−" : " "}${m.toFixed(16)} × 2<sup>${E}</sup>`;
}

// the left-hand equation panel never changes, so render it once
function renderTex() {
  for (const el of document.querySelectorAll("[data-tex]")) {
    katex.render(el.dataset.tex, el);
  }
}

const M_MAX = 55;
const COLORS = { x: "#2a78d6", x_hi: "#1baf7a", x_lo: "#e34948" };  // blue, green, red

// Highest and lowest bit position reached over every M, so the columns stay
// put while you drag the slider. splitAt(M) returns { values, cuts }.
function bitRange(splitAt) {
  let hi = -Infinity, lo = Infinity;
  for (let M = 0; M <= M_MAX; M++) {
    const { values, cuts } = splitAt(M);
    for (const cut of cuts) hi = Math.max(hi, cut);
    for (const v of values) {
      for (const [p] of significandBits(v)) {
        hi = Math.max(hi, p);
        lo = Math.min(lo, p);
      }
    }
  }
  return [hi + 1, lo];  // one spare column on the left for sign markers
}

// How many of x's 53 bit positions are at or above the cut (x_hi's side).
function hiBitCount(x, cut) {
  if (x === 0) return 0;
  const top = significandBits(x)[0][0];
  return Math.min(Math.max(top - cut + 1, 0), 53);
}

// Draw x = x_hi + x_lo as three stacked panels of bit rows, with = and +
// between them. Columns line up across panels, so bits can be compared.
// panels: [{ op, color, rows: [{ name, value, cut, note }] }]: op is the
//   symbol drawn above the panel ("", "=", "+"); cut is that row's cutPower,
//   drawn as a dashed line; note (optional) is right-aligned text after the row.
// header: { cut, text }, a label above everything centered on that cut.
// Filled = 1, outlined = 0, nothing = no bit there. Higher powers on the left.
function drawSplit(el, panels, [hi, lo], header) {
  const LABEL = 70, WIDTH = 900, ROW = 30, PAD = 6, OP = 30, HEAD = 30;
  const hasNotes = panels.some(panel => panel.rows.some(row => row.note));
  const plotRight = WIDTH - (hasNotes ? 90 : 0);
  const colW = (plotRight - LABEL) / (hi - lo + 1);
  const colX = p => LABEL + (hi - p) * colW;  // left edge of column p
  // the cut sits between column `cut` (kept) and column `cut - 1` (rounded away)
  const cutX = cut => colX(cut - 1);

  const headerX = Math.min(Math.max(cutX(header.cut), 170), plotRight - 170);  // keep it on screen
  let svg = `<text x="${headerX}" y="${HEAD - 12}" text-anchor="middle">${header.text}</text>`;
  let y = HEAD;

  panels.forEach(({ op, color, rows }, i) => {
    if (i > 0) {
      svg += `<text x="${LABEL / 2}" y="${y + OP / 2 + 7}" text-anchor="middle" font-size="22">${op}</text>`;
      y += OP;
    }
    const panelHeight = 2 * PAD + rows.length * ROW;
    svg += `<rect x="0.5" y="${y}" width="${WIDTH - 1}" height="${panelHeight}" rx="6"
              fill="none" stroke="${color}" stroke-opacity="0.5"/>`;

    rows.forEach(({ name, value, cut, note }, r) => {
      const rowY = y + PAD + r * ROW;
      const textY = rowY + ROW / 2 + 4;
      svg += `<text x="8" y="${textY}">${name}</text>`;
      svg += `<rect x="${LABEL}" y="${rowY + 4}" width="${plotRight - LABEL}" height="${ROW - 8}" fill="#f0f0f0"/>`;

      const bits = significandBits(value);
      for (const [p, b] of bits) {
        svg += `<rect x="${colX(p) + 0.5}" y="${rowY + 4}" width="${colW - 1}" height="${ROW - 8}"
                  fill="${b ? color : "white"}" stroke="${color}">
                  <title>${name}: bit at 2^${p} = ${b}</title></rect>`;
      }
      // the bars only show magnitude, so mark the sign next to the leading bit
      if (bits.length) {
        svg += `<text x="${colX(bits[0][0]) - 2}" y="${textY}" text-anchor="end">${value < 0 ? "−" : "+"}</text>`;
      }
      // per-row segments: equal cuts in neighbouring rows join into one line
      svg += `<line x1="${cutX(cut)}" x2="${cutX(cut)}" y1="${rowY}" y2="${rowY + ROW}"
                stroke="#222" stroke-width="2" stroke-dasharray="4 3"/>`;
      if (note) svg += `<text x="${WIDTH - 8}" y="${textY}" text-anchor="end">${note}</text>`;
    });
    y += panelHeight;
  });

  // label every 4th column with its power of two
  const height = y + 30;
  for (let p = hi; p >= lo; p--) {
    if (p % 4 !== 0) continue;
    svg += `<text class="tick" x="${colX(p) + colW / 2}" y="${height - 8}">2<tspan dy="-5" font-size="8">${p}</tspan></text>`;
  }

  el.setAttribute("viewBox", `0 0 ${WIDTH} ${height}`);
  el.innerHTML = svg;
}
