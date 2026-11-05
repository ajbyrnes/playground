# GPUs covered in (or relevant to) the WCP / talk

Fill in peak dense throughput (no sparsity) to get the low-precision : FP64 ratio.
For NVIDIA, FP64 has two numbers on many parts: vector (CUDA cores) and Tensor Core. Note which one the ratio uses.
Ootomo et al.'s ratios on the slides use FP64 Tensor Core where it exists (A100) and FP64 vector otherwise.

## Used in the papers' evaluations

| GPU | Vendor | Arch | Where it appears | FP64 vector | FP64 matrix/TC | INT8 dense | FP8 dense | Ratio |
|---|---|---|---|---|---|---|---|---|
| TITAN RTX | NVIDIA | Turing | Mukunoki et al. (2020); Ootomo et al. (2024) | | — | | — | |
| V100 | NVIDIA | Volta | Mukunoki et al. (2020); intro FP64 table | | — | — (no INT8 TC) | — | |
| A100 (40GB SXM4) | NVIDIA | Ampere | Ootomo et al. (2024); intro FP64 table | | | | — | |
| RTX A6000 | NVIDIA | Ampere | Ootomo et al. (2024) | | — | | — | |
| RTX 6000 Ada | NVIDIA | Ada Lovelace | Ootomo et al. (2024) | | — | | | |
| GB200 (per B200 GPU) | NVIDIA | Blackwell | Schwarz et al. (2026) | 40 | 40 | 5,000 | 5,000 | 1 : 125 (INT8 dense : FP64) |
| RTX Pro 6000 Blackwell Server Edition | NVIDIA | Blackwell | Schwarz et al. (2026) | 1.875* | — | ~960* | ~960* | 1 : 512* (INT8 dense : FP64) |
| B200 (DGX B200) | NVIDIA | Blackwell | Brower et al. (2026); intro FP64 table | | | | | |

## Mentioned in the WCP paper / talk, not evaluated

| GPU | Vendor | Arch | Where it appears | FP64 vector | FP64 matrix/TC | INT8 dense | FP8 dense | Ratio |
|---|---|---|---|---|---|---|---|---|
| H100 | NVIDIA | Hopper | Intro FP64 table | | | | | |
| B300 | NVIDIA | Blackwell Ultra | Intro FP64 table (FP64 cut to 1.3 TFLOP/s) | | | | | |
| GH200 | NVIDIA | Hopper | TOP500 #5 (JUPITER Booster) | | | | | |
| MI250X | AMD | CDNA 2 | TOP500 #3 (Frontier); vendor-loop slide | | | | — | |
| MI300A | AMD | CDNA 3 | TOP500 #2 (El Capitan); vendor-loop slide | | | | | |

## Not in the paper, but relevant to the hardware / vendor-loop argument

| GPU | Vendor | Arch | Why relevant | FP64 vector | FP64 matrix/TC | INT8 dense | FP8 dense | Ratio |
|---|---|---|---|---|---|---|---|---|
| H200 | NVIDIA | Hopper | Last Hopper datacenter part; same FP64 as H100 | | | | | |
| GB300 (per B300 GPU) | NVIDIA | Blackwell Ultra | Rack-scale version of B300 | | | | | |
| Rubin (VR200) | NVIDIA | Rubin | Next generation; does the FP64 trend continue? | | | | | |
| MI300X | AMD | CDNA 3 | AMD's AI-oriented sibling of the MI300A | | | | | |
| MI325X | AMD | CDNA 3 | Refresh of MI300X | | | | | |
| MI355X | AMD | CDNA 4 | Current AMD datacenter part; check whether AMD is trending the same way as NVIDIA | | | | | |

Notes to check while filling this in:
- Whether B300 / Blackwell Ultra also reduced INT8 (not only FP64). That bears on the "Is INT8 support here to stay?" point in the conclusion.
- Whether AMD's newest parts (MI355X) keep FP64 matrix throughput or trade it for FP8/FP4, which bears on the "vendor loop" slide.
- Intel Data Center GPU Max (Aurora, TOP500 #4) is the only non-NVIDIA/AMD accelerator in the TOP500 table, if you want it for completeness.

## Sources and derivations (filled 2026-11-04)

Units: TFLOPS for FP64, dense TOPS/TFLOPS for INT8/FP8.

**GB200 (per GPU).** NVIDIA GB200 NVL72 spec table (https://www.nvidia.com/en-us/data-center/gb200-nvl72/), rack column, 72 GPUs:
- FP64 / FP64 Tensor Core: 2,880 TFLOPS → 40 per GPU (no sparsity footnote on this row; vector and Tensor Core given as one number)
- INT8 Tensor Core: 720 POPS, footnote "Specification in sparse. Dense is one-half sparse spec shown." → 360 POPS dense → 5,000 TOPS per GPU
- FP8/FP6 Tensor Core: 720 PFLOPS, same footnote → 5,000 TFLOPS dense per GPU
- Ratio: 5,000 / 40 = 125

**RTX Pro 6000 Blackwell Server Edition (* = derived, not published).** NVIDIA's datasheet (doc 3850600, Apr25) lists only FP32 = 120 TFLOPS and "Peak FP4 AI PFLOPS" = 4 PFLOPS. No FP64 or INT8 row. Derived from the GB202 chip it uses, per NVIDIA's RTX Blackwell GPU architecture whitepaper (https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf):
- "The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations." → 120 / 64 = 1.875 TFLOPS
- RTX 5090 (also GB202) table: Peak FP32 104.8 TFLOPS, Peak INT8 Tensor 838 TOPS dense (1,676 sparse) → INT8 dense = 8 × FP32 → 8 × 120 ≈ 960 TOPS
- Consistency check: on the 5090, FP4 sparse = 4 × INT8 dense (3,352 vs 838). For the Pro 6000, 4 × 960 = 3,840, which matches the datasheet's "4 PFLOPS".
- Ratio: 8 × 64 = 512. This doesn't depend on clock speed, only on the GB202 per-SM rates.
