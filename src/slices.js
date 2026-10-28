// Shared by split_vector.html, split_mul.html and eft_mul.html: splitting
// vectors and matrices into slices with M = β, seeded test data, an exact
// check, and the drawings. Needs bits.js (significandBits).

// ---- splitting (mirrors splitting.py) ----

// Equation (6) with u = 2^-53: β takes the place of M, and each slice then
// holds α = 53 - β bits.
function betaFor(n) {
  return Math.ceil((53 + Math.log2(n)) / 2);
}

// splitting.py's split_vector with M = β. Each pass is ExtractVector on the
// remainder r, with σ from r's own max.
function splitVector(x, beta) {
  const passes = [];
  let r = x;
  while (r.some(v => v !== 0)) {
    const e = Math.ceil(Math.log2(Math.max(...r.map(Math.abs))));
    const sigma = 2 ** beta * 2 ** e;
    const slice = r.map(v => sigma + v - sigma);
    r = r.map((v, i) => v - slice[i]);
    // the slice's window: from 2^e down to the last bit fl(σ + r) can hold
    passes.push({ e, cut: e + beta - 52, slice });
  }
  return passes;
}

// splitting.py's split_mat (Algorithm 3). Each row has its own σ, so rows
// split independently: row i of D[k] is row i's k-th splitVector slice, or
// zeros once that row has run out. rows[i] keeps row i's passes for drawing.
function splitMat(A, beta) {
  const rows = A.map(row => splitVector(row, beta));
  const K = Math.max(...rows.map(passes => passes.length));
  const D = Array.from({ length: K }, (_, k) =>
    rows.map((passes, i) => passes[k] ? passes[k].slice : A[i].map(() => 0)));
  return { rows, D };
}

const transpose = M => M[0].map((_, j) => M.map(row => row[j]));

// fl(A B), one dot product per entry
function matmul(A, B) {
  return A.map(row => B[0].map((_, j) => row.reduce((sum, a, k) => sum + a * B[k][j], 0)));
}

// ---- exact arithmetic, for checking ----

// v = mantissa · 2^exp exactly, with the mantissa an integer (a BigInt).
function exactParts(v) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, v);
  const bits = view.getBigUint64(0);
  const field = Number((bits >> 52n) & 0x7FFn);
  let mantissa = bits & ((1n << 52n) - 1n);
  if (field !== 0) mantissa |= 1n << 52n;  // hidden bit (subnormals have none)
  const exp = (field === 0 ? 1 : field) - 1075;
  return [bits >> 63n ? -mantissa : mantissa, exp];
}

// Doubles and products of two doubles as integer multiples of 2^-2200: the
// smallest product of two doubles is 2^-2148, so this never loses a bit, and
// sums of these are exact.
function exact(v) {
  const [m, e] = exactParts(v);
  return m << BigInt(e + 2200);
}
function exactProduct(a, b) {
  const [ma, ea] = exactParts(a), [mb, eb] = exactParts(b);
  return (ma * mb) << BigInt(ea + eb + 2200);
}

// ---- seeded test data ----

// Small seeded RNG (mulberry32): the same seed gives the same significands
// at every span, so a chart's curve is for one fixed set of numbers.
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

// ±(1 + fraction) with all 52 fraction bits random: two draws, since one
// only has 32 random bits
function randomSignificand(rng) {
  const sign = rng() < 0.5 ? -1 : 1;
  const fraction = (Math.floor(rng() * 2 ** 26) * 2 ** 27 + Math.floor(rng() * 2 ** 27)) / 2 ** 53;
  return sign * (1 + fraction);
}

// exponent of position i out of n when spreading span bits evenly around 0:
// from ⌈span/2⌉ down to ⌈span/2⌉ - span, so max - min is exactly span
const spreadExponent = (i, n, span) => Math.ceil(span / 2) - Math.round(i * span / (n - 1));

// ---- drawing ----

// Which slice a bit at power p of an element belongs to, given that
// element's splitVector passes. Usually the window it falls in. A bit in the
// gap between window k's cut and window k+1's top goes to slice k: slice k
// absorbed it, either as 0s it skipped or through rounding up. Trailing 0s
// below the last cut go to the last slice. Returns [slice, inGap].
function sliceOf(passes, p) {
  const k = passes.findIndex(q => q.cut <= p);
  if (k < 0) return [passes.length - 1, true];
  if (p > passes[k].e) return [Math.max(k - 1, 0), true];
  return [k, false];
}

// A distinct color for every slice, however many there are (300 bits of
// span is ~15 slices). Each hue steps by the golden angle (~137.5°), so
// consecutive slices land far apart on the color wheel and later ones fill
// the gaps; cycling the lightness separates hues that end up close together.
// Slice 1 starts at blue, like x elsewhere.
function sliceColor(k) {
  const hue = (212 + k * 137.508) % 360;
  const lightness = [48, 64, 36][k % 3];
  return `hsl(${hue.toFixed(1)}, 70%, ${lightness}%)`;
}

// Bits colored by slice window. One block per group, one row per value:
// groups is [{ label, names, values, passes, note }], where passes is that
// group's splitVector result (a vector is one group; a matrix, one per row).
// label is drawn left of the block, note right of it.
function drawBands(el, groups) {
  const total = groups.reduce((sum, g) => sum + g.values.length, 0);
  const ROW = total > 16 ? 12 : 22, GAP = 12, TOP = 22;
  const LABEL = 110, NOTE = groups.some(g => g.note) ? 70 : 0, WIDTH = 900;
  const plotRight = WIDTH - NOTE;
  const allTops = groups.flatMap(g => g.values.filter(v => v !== 0).map(v => significandBits(v)[0][0]));
  const hi = Math.max(...groups.map(g => g.passes[0].e)) + 1;  // spare column for sign markers
  const lo = Math.min(...allTops) - 52;
  const colW = (plotRight - LABEL) / (hi - lo + 1);
  const colX = p => LABEL + (hi - p) * colW;  // left edge of column p

  let svg = "", y = TOP;
  groups.forEach(({ label, names, values, passes, note }, g) => {
    const blockTop = y, blockBottom = y + values.length * ROW;

    // this block's band backgrounds, numbered above the first block only
    passes.forEach(({ e, cut }, k) => {
      const top = Math.min(e, hi), low = Math.max(cut, lo);
      if (top < low) return;
      const x0 = colX(top), w = (top - low + 1) * colW;
      svg += `<rect x="${x0}" y="${blockTop - 3}" width="${w}" height="${blockBottom - blockTop + 6}"
                fill="${sliceColor(k)}" fill-opacity="0.08"/>`;
      if (g === 0 && w > 14) {
        svg += `<text x="${x0 + w / 2}" y="${blockTop - 8}" text-anchor="middle" fill="${sliceColor(k)}">${k + 1}</text>`;
      }
      // the cut: between column `cut` (kept) and column `cut - 1` (next slice),
      // skipped when it falls past the lowest bit drawn
      if (cut - 1 >= lo) {
        svg += `<line x1="${colX(cut - 1)}" x2="${colX(cut - 1)}" y1="${blockTop - 3}" y2="${blockBottom + 3}"
                  stroke="#222" stroke-width="1" stroke-dasharray="3 3"/>`;
      }
    });

    const midY = (blockTop + blockBottom) / 2 + 4;
    if (label) svg += `<text x="0" y="${midY}">${label}</text>`;
    if (note) svg += `<text x="${WIDTH - 4}" y="${midY}" text-anchor="end">${note}</text>`;

    values.forEach((v, i) => {
      if (v === 0) return;
      const rowY = blockTop + i * ROW, textY = rowY + ROW / 2 + 4;
      const top = significandBits(v)[0][0];
      if (ROW >= 18) {
        // a block label takes the name's place (the hover text still has it)
        if (names && !label) svg += `<text x="0" y="${textY}">${names[i]}</text>`;
        svg += `<text x="${LABEL - 8}" y="${textY}" text-anchor="end">2<tspan dy="-5" font-size="8">${top}</tspan></text>`;
      }
      for (const [p, b] of significandBits(v)) {
        const [k, inGap] = sliceOf(passes, p);
        svg += `<rect x="${colX(p)}" y="${rowY + 2}" width="${colW}" height="${ROW - 4}"
                  fill="${sliceColor(k)}" fill-opacity="${b ? 1 : 0.25}">
                  <title>${names ? names[i] : ""}: bit at 2^${p} = ${b}, slice ${k + 1}${inGap ? " (outside its window, absorbed by rounding)" : ""}</title></rect>`;
      }
      svg += `<text x="${colX(top) - 2}" y="${textY}" text-anchor="end">${v < 0 ? "−" : "+"}</text>`;
    });

    y = blockBottom + GAP;
    if (g < groups.length - 1) {
      svg += `<line x1="0" x2="${WIDTH}" y1="${y - GAP / 2}" y2="${y - GAP / 2}" stroke="#ddd"/>`;
    }
  });

  // a power-of-two tick every `step` columns, spaced at least ~30px apart
  const step = [4, 8, 16, 32, 64].find(s => s * colW >= 30) || 64;
  const height = y + 24;
  for (let p = hi; p >= lo; p--) {
    if (p % step !== 0) continue;
    svg += `<text class="tick" x="${colX(p) + colW / 2}" y="${height - 8}">2<tspan dy="-5" font-size="8">${p}</tspan></text>`;
  }

  el.setAttribute("viewBox", `0 0 ${WIDTH} ${height}`);
  el.innerHTML = svg;
}

// Step chart of counts against span 0..xMax. series: [{ ys, color, dash,
// label }] with ys[span]; markers: [{ x, y, color }] for the current spans.
function drawStepChart(el, { xMax, series, markers, xLabel, yLabel }) {
  const W = 900, H = 240, L = 50, R = 16, T = 16, B = 40;
  const yMax = Math.max(...series.flatMap(s => s.ys)) + 1;
  const px = s => L + (s / xMax) * (W - L - R);
  const py = k => H - B - (k / yMax) * (H - T - B);
  // hold each value until the next span
  const stepPath = ys => ys.map((k, s) => `${s ? "H" + px(s) : "M" + px(0)} V${py(k)}`).join(" ");

  let svg = "";
  const yStep = yMax > 20 ? 4 : 2;
  for (let k = 0; k <= yMax; k += yStep) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${py(k)}" y2="${py(k)}" stroke="#eee"/>`;
    svg += `<text x="${L - 6}" y="${py(k) + 4}" text-anchor="end">${k}</text>`;
  }
  for (let s = 0; s <= xMax; s += 50) {
    svg += `<text x="${px(s)}" y="${H - B + 16}" text-anchor="middle">${s}</text>`;
  }
  svg += `<line x1="${L}" x2="${W - R}" y1="${py(0)}" y2="${py(0)}" stroke="#999"/>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${xLabel}</text>`;
  svg += `<text x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">${yLabel}</text>`;

  series.forEach(({ ys, color, dash, label }, i) => {
    svg += `<path d="${stepPath(ys)}" fill="none" stroke="${color}" stroke-width="${dash ? 1.5 : 2}"
              ${dash ? `stroke-dasharray="5 4"` : ""}/>`;
    // legend, top left
    const ly = T + 8 + i * 18;
    svg += `<line x1="${L + 12}" x2="${L + 36}" y1="${ly}" y2="${ly}" stroke="${color}" stroke-width="2"
              ${dash ? `stroke-dasharray="5 4"` : ""}/>`;
    svg += `<text x="${L + 42}" y="${ly + 4}">${label}</text>`;
  });

  for (const { x, y, color } of markers) {
    svg += `<line x1="${px(x)}" x2="${px(x)}" y1="${T}" y2="${py(0)}" stroke="${color}" stroke-dasharray="2 3"/>`;
    svg += `<circle cx="${px(x)}" cy="${py(y)}" r="4.5" fill="${color}" stroke="white" stroke-width="1.5"/>`;
  }

  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.innerHTML = svg;
}

// A matrix as a small grid, one cell per entry. fill(i, j) gives each cell's
// color (and fill-opacity, as [color, opacity]); title(i, j) its hover text.
function tile(M, fill, caption, title, above = "") {
  const CELL = 20;
  let cells = "";
  M.forEach((row, i) => row.forEach((v, j) => {
    const [color, opacity] = [].concat(fill(i, j), 1);
    cells += `<rect x="${j * CELL}" y="${i * CELL}" width="${CELL - 1}" height="${CELL - 1}"
                fill="${color}" fill-opacity="${opacity}"><title>${title(i, j)}</title></rect>`;
  }));
  const w = M[0].length * CELL, h = M.length * CELL;
  return `<figure class="tile"><div class="scale">${above}</div><svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${cells}</svg>
            <figcaption>${caption}</figcaption></figure>`;
}

// The powers of two a matrix's bits span: from the largest leading bit of
// any entry down to the lowest 1 bit of any entry, e.g. "2^0 … 2^-24".
function scaleLabel(M) {
  const bits = M.flat().filter(v => v !== 0).map(significandBits);
  if (!bits.length) return "0";
  const top = Math.max(...bits.map(b => b[0][0]));
  const low = Math.min(...bits.map(b => b.filter(([, bit]) => bit).at(-1)[0]));
  const pow = p => `2<sup>${p}</sup>`;
  return top === low ? pow(top) : `${pow(top)} … ${pow(low)}`;
}

// M = S(1) + S(2) + ... as tiles: M shaded by exponent (darker = larger),
// each slice's nonzero entries in its slice color. name is the slices' name
// ("D" or "E"), matrixName the matrix's ("A" or "B").
function sliceTiles(name, M, slices, matrixName) {
  const exps = M.flat().filter(v => v !== 0).map(v => significandBits(v)[0][0]);
  const eMax = Math.max(...exps), eMin = Math.min(...exps);
  const shade = v => v === 0 ? 0 : 0.15 + 0.85 * (significandBits(v)[0][0] - eMin) / Math.max(eMax - eMin, 1);

  let html = tile(M, (i, j) => ["#555", shade(M[i][j])], matrixName,
    (i, j) => `${matrixName}[${i}][${j}] = ${M[i][j]}`, scaleLabel(M));
  slices.forEach((S, k) => {
    html += `<span class="op">${k ? "+" : "="}</span>`;
    html += tile(S, (i, j) => S[i][j] === 0 ? "#eee" : sliceColor(k), `${name}<sup>(${k + 1})</sup>`,
      (i, j) => `${name}(${k + 1})[${i}][${j}] = ${S[i][j]}`, scaleLabel(S));
  });
  return html;
}
