"""Extract every figure and table (as an image) from the four subject papers,
with their captions.

Run from the repo root:
    uv run --with pymupdf python figures/scripts/paper_figures.py

Writes figures/papers/<paper>/<fig01|table01|...>.png, plus
figures/papers/captions.md (a browsable index) and captions.json.

Regions are found automatically: figures between their caption and the
nearest body paragraph above, tables between their caption and the nearest
body paragraph below. Where that heuristic gets a layout wrong (side
captions, page headers, stray text), RECTS overrides the region by hand, and
CAPTIONS replaces captions whose math didn't survive text extraction.
"""
import json
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "papers"
OUT = ROOT / "figures" / "papers"
DPI = 250

SOURCES = {
    "ozaki": ("Error-free_transformations_of_matrix_multiplication.pdf",
              "Ozaki et al., Error-free transformations of matrix multiplication (2012)"),
    "ootomo": ("DGEMM_on_integer_matrix_multiplication_unit.pdf",
               "Ootomo et al., DGEMM on integer matrix multiplication unit"),
    "schwarz": ("Guaranteed_DGEMM_Accuracy_While_Using_Reduced_Precision_Tensor_Cores_Through_Extensions_of_the_Ozaki_Scheme.pdf",
                "Schwarz et al., Guaranteed DGEMM accuracy while using reduced precision Tensor Cores"),
    "brower": ("Mixed-Precision_Ab_Initio_Tensor_Network_State_Methods_Adapted_for_NVIDIA_Blackwell_Technology_via_Emulated_FP64_Arithmetic.pdf",
               "Brower et al., Mixed-precision ab initio tensor network state methods on NVIDIA Blackwell"),
}

# Hand-set regions, in PDF points [x0, y0, x1, y1] on the caption's page.
RECTS = {
    "ozaki": {  # Springer side captions: the figure sits to the right of its caption
        "fig01": [155, 50, 391, 222], "fig02": [48, 480, 391, 571], "fig03": [155, 444, 391, 612],
        "fig04": [48, 310, 391, 578], "fig05": [152, 50, 391, 212],
        "table03": [48, 65, 391, 124], "table04": [164, 50, 391, 110], "table05": [48, 559, 391, 609],
    },
    "ootomo": {  # mostly trimming the running-header rule
        "fig01": [300, 56, 546, 202], "fig03": [58.8, 56, 545.9, 234.7], "fig07": [58.8, 56, 545.9, 341.1],
        "fig08": [49.3, 56, 536.5, 224.4], "fig10": [58.8, 56, 545.9, 259.2], "fig11": [294.6, 56, 536.5, 220.6],
        "table01": [58, 86, 546, 197], "table02": [58.8, 95.3, 299.4, 206],
    },
    "schwarz": {"fig05": [306, 70.6, 561.2, 212], "fig06": [47.8, 71, 562.4, 464]},
    "brower": {"fig01": [311.5, 54, 570.4, 242.1], "fig03": [48.5, 128, 306.7, 335.4],
               "fig04": [42.3, 54, 306.7, 251.3]},
}

# Captions whose math or layout extraction mangled, rewritten by hand from the PDFs.
CAPTIONS = {
    "ozaki": {
        "fig01": "Fig. 1 Each rectangle depicts a floating-point number. Left and right end points of a rectangle show the unit in the first place and in the last place, respectively. Algorithm 2 divides floating-point numbers into two floating-point numbers",
        "fig02": "Fig. 2 Assume that vectors $x$ and $y$ are divided into $x^{(1)} + x^{(2)}$ and $y^{(1)} + y^{(2)}$, respectively. A floating-point evaluation of $(x^{(1)})^T y^{(1)}$ does not cause rounding errors. It is expected that the magnitude of $(x^{(1)})^T y^{(1)}$ is greater than the magnitude of $(x^{(2)})^T y^{(1)}$ and $(x^{(2)})^T y$",
        "fig03": "Fig. 3 This visualizes the dot product. Each of $x^{(1)}_1 y^{(1)}_1, \\ldots, x^{(1)}_n y^{(1)}_n$ has at most $2\\alpha$ nonzero leading bits, so that $|(x^{(1)})^T y^{(1)}|$ is representable in at most $-\\log_2 \\mathbf{u}$ bits",
        "fig04": "Fig. 4 In this figure the number of nonzero leading bits in $(x^{(i)})^T y^{(j)}$ $(i + j \\le k)$ does not exceed $-\\log_2 \\mathbf{u}$. The computation of $\\mathrm{fl}((x^{(i)})^T y^{(j)})$ may cause rounding errors only for $i + j = k + 1$",
        "fig05": "Fig. 5 This figure illustrates the splittings. Each floating-point number has at most $\\alpha$ nonzero leading bits",
        "table04": "Table 4 Comparison of the computing times",
    },
    "ootomo": {
        "fig01": "Figure 1. The basic computation of high-precision matrix multiplication $C \\leftarrow AB$ algorithms on low-precision computing unit.",
        "fig02": "Figure 2. Calculation of the mantissa part in multiplication and addition of two floating-point values, $v_1$ and $v_2$.",
        "fig03": "Figure 3. The comparison of the elementwise-place splitting (Left) and shared-place splitting (Right) methods for splitting the vector $\\mathbf{a} = [a_1\\, a_2 \\cdots a_k]$ into several vectors $\\mathbf{a}^{(\\cdot)}$. $e_i$ is the exponent of $a_i$. The same applies to the vector $\\mathbf{b}$.",
        "fig07": "Figure 7. The accuracy evaluation of matrix multiplications where the elements of input matrices have various exponent ranges ($\\phi$ = 0.1, 1, 2 and 4).",
        "fig08": "Figure 8: The accuracy comparison of matrix multiplication of a matrix and its approximate inverse $AA^\\dagger$.",
    },
    "schwarz": {
        "fig01": "Figure 1: Unsigned slice encoding using two's complement arithmetic. Case 1: u8 values in [0,127] map directly to s8 without modification. Case 2: u8 values in [128,255] are remapped as $256 - x$ with a +256 carry to the higher slice, while storing $-x$ in s8. Bit patterns are preserved, e.g., 200 (u8) ≡ −56 (s8) = 0b11001000.",
        "fig02": "Figure 2: ADP-enabled DGEMM, configured with six distinct mantissa bit counts for the Ozaki-I algorithm, on Test 2, where $n = 1024$. For each mantissa bit count, we display the variant without the option to fall back to native FP64 DGEMM (solid lines) and the variant with guardrails and automatic fallback to native FP64 DGEMM (dashed lines).",
        "fig03": "Figure 3: Maximum componentwise relative error when multiplying two random uniformly distributed matrices.",
        "fig04": "Figure 4: Average componentwise relative error when multiplying two random uniformly distributed matrices.",
        "fig05": "Figure 5: Breakdown of DGEMM performance when emulating 55 mantissa bits on NVIDIA Tensor Cores. For this experiment, ADP is forced to always use 55 bits, regardless of input characteristics, in order to maximize its relative overhead (worst-case configuration).",
    },
    "brower": {
        "fig01": "Figure 1. Relative error of the ground state energy as a function of DMRG iteration steps for the F$_2$ molecule in a CAS(18,18) model space using $D_{SU(2)} = 1024$ (left panel) and $D_{SU(2)} = 8192$ (right panel) SU(2) multiplets for the nonemulated native FP64 limit and for various number of INT8 slices, $\\kappa \\in \\{6,4,3,2\\}$ obtained on a DGX B200 system. The dashed line represents the relative error of chemical accuracy.",
        "fig02": "Figure 2. Similar to Figure 1 but for the nitrogen dimer at its equilibrium geometry in a CAS(14,28) model space using $D_{SU(2)} = 1024$ (left panel) and $D_{SU(2)} = 4096$ (right panel) SU(2) multiplets obtained on a DGX B200 system.",
        "fig03": "Figure 3. Shifted ground state energy for the spin-1/2 doublet state of the cytochrome P450 (CYP) enzymes with CAS(63,58) model space as a function of DMRG iteration using $D_{SU(2)} = 2048$ SU(2) multiplets and $\\kappa \\in \\{4,6\\}$ slices (left panel) and the absolute error measured with respect to the native FP64 data sets for $\\kappa = 4$ (central panel) and $\\kappa = 6$ (right panel) slices for various $D$ values obtained on a DGX B200 system.",
        "fig05": "Figure 5. Benchmark results obtained via the SU(2) spin-adapted single node hybrid CPU plus multi-GPU DMRG calculations for the F$_2$ molecule on a CAS(18,18) orbital space, the N$_2$ molecule on a CAS(14,28) space, FeMoco on CAS(54,54) and CAS(113,76) spaces, and P450 on CAS(63,58). The solid lines correspond to calculations performed on a DGX B200 system via native FP64 precision, while dashed lines correspond to the emulated performant mode. As a reference, the dotted lines trace the results obtained on a DGX H100 system. Numbers indicate the corresponding U(1) bond dimension values, which are the same for the dotted, dashed, and solid lines.",
    },
}

CAP = re.compile(r"^(Fig\.?|Figure|Table)\s*(\d+)\s*[.:]?", re.I)
# "Figure 2 shows ..." in body text is a reference, not a caption
REF = re.compile(r"^(Fig\.?|Figure|Table)\s*\d+\s+[a-z]")


def is_body(b):
    t = b[4].strip()
    return len(t) > 160 and len(t) / (t.count("\n") + 1) > 35


def find(doc):
    """[(id, page index, Rect, caption)] for every caption in doc."""
    found = {}
    for pno, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        blocks = [b for b in page.get_text("blocks") if b[4].strip()]
        caps = []
        for b in blocks:
            t = b[4].strip()
            m = CAP.match(t)
            if m and not (REF.match(t) and not t.startswith("Fig. ")):
                kind = "table" if m.group(1).lower().startswith("tab") else "fig"
                caps.append((f"{kind}{int(m.group(2)):02d}", kind, b))
        for fid, kind, cb in caps:
            if fid in found:
                continue
            x0, y0, x1, y1 = cb[:4]
            if x1 - x0 > 0.55 * W:
                cx0, cx1 = 40, W - 40
            elif x0 > W / 2 - 10:
                cx0, cx1 = W / 2, W - 40
            else:
                cx0, cx1 = 40, W / 2
            incol = [b for b in blocks if b[0] < cx1 and b[2] > cx0 and b is not cb]
            text, cy1 = cb[4].strip(), y1
            for b in sorted(incol, key=lambda b: b[1]):
                if 0 <= b[1] - cy1 < 3 and abs(b[0] - x0) < 5 and not CAP.match(b[4].strip()):
                    text += " " + b[4].strip()
                    cy1 = b[3]
            stop = lambda b: is_body(b) or CAP.match(b[4].strip())
            if kind == "fig":
                top = max([b[3] for b in incol if b[3] <= y0 + 1 and stop(b)], default=50)
                rect = fitz.Rect(cx0, top + 2, cx1, y0 - 1)
            else:
                bot = min([b[1] for b in incol if b[1] >= cy1 - 1 and stop(b)], default=H - 55)
                rect = fitz.Rect(cx0, cy1 + 1, cx1, bot - 2)
            # shrink to the ink inside (drawings, images, text wholly within),
            # which drops running headers and margins the heuristic swept up
            ink = [fitz.Rect(b[:4]) for b in blocks if b is not cb and fitz.Rect(b[:4]) in rect + (-3, -3, 3, 3)]
            ink += [dr["rect"] for dr in page.get_drawings() if dr["rect"].intersects(rect) and dr["rect"].width < W * 0.95]
            ink += [fitz.Rect(im["bbox"]) for im in page.get_image_info() if fitz.Rect(im["bbox"]).intersects(rect)]
            if ink:
                u = fitz.Rect(ink[0])
                for r in ink[1:]:
                    u |= r
                rect = (u & rect) + (-3, -3, 3, 3)
            found[fid] = (pno, rect, " ".join(text.split()))
    return found


def autocrop(pix, pad=12):
    """Trim the white margin around the ink in a rendered region."""
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    ink = np.argwhere(a.min(axis=2) < 235)
    if not len(ink):
        return None
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    s = 72 / DPI  # pixels back to points
    return fitz.Rect(c0 - pad, r0 - pad, c1 + pad, r1 + pad) * s


def main():
    index = {}
    for name, (pdf, cite) in SOURCES.items():
        doc = fitz.open(PAPERS / pdf)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        entries = []
        for fid, (pno, rect, caption) in sorted(find(doc).items(), key=lambda kv: (kv[0][0] == "t", kv[0])):
            rect = fitz.Rect(RECTS.get(name, {}).get(fid, rect))
            page = doc[pno]
            trim = autocrop(page.get_pixmap(clip=rect, dpi=DPI))
            if trim:
                rect = fitz.Rect(trim.x0 + rect.x0, trim.y0 + rect.y0,
                                 trim.x1 + rect.x0, trim.y1 + rect.y0) & rect
            page.get_pixmap(clip=rect, dpi=DPI).save(OUT / name / f"{fid}.png")
            entries.append(dict(id=fid, file=f"{name}/{fid}.png", page=pno + 1,
                                caption=CAPTIONS.get(name, {}).get(fid, caption)))
        index[name] = dict(source=cite, figures=entries)
        print(f"{name}: {len(entries)} items")

    (OUT / "captions.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    md = ["# Figures and tables from the subject papers", "",
          "Generated by `figures/scripts/paper_figures.py`. Paths are relative to this file.", ""]
    for name, v in index.items():
        md += [f"## {name}: {v['source']}", ""]
        for e in v["figures"]:
            md += [f"### `{e['file']}` (p. {e['page']})", "", f"![]({e['file']})", "", f"> {e['caption']}", ""]
    (OUT / "captions.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
