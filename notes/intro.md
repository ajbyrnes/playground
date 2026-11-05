# Intro

Sources are the six articles compiled in `papers/Ozaki_context.pdf` (HPCwire and The Next Platform, April 2025 to June 2026).

## FP64 on new GPUs

The TOP500 ranks supercomputers by double-precision LINPACK, and FP64 is still how the HPC community measures itself. But GPU design is now driven by AI, and AI wants low precision. The table shows what that's done to FP64 on NVIDIA's data-center GPUs: big jumps through Hopper, then flat with Blackwell, and on the B300 it's essentially gone (1.3 TFLOP/s). Rubin brings native FP64 back to 33 TFLOP/s, which is still below Blackwell (B200 37, GB200 40) and about the same as the H100 from 2022.

Jack Dongarra said it at the TOP500 press conference at SC25: "The floating-point capability of the platform is not improved – not improved – over the previous generation." (Quoted in Woodie, HPCwire, Feb 2026.)

NVIDIA's answer is emulation. The 200 TFLOP/s in Rubin's Tensor Core column is FP64 matrix throughput *with* the cuBLAS emulation turned on, not native hardware; the matching emulated figure for Blackwell is 150 TFLOP/s. NVIDIA's argument (January 2026 blog post, quoted in both Woodie articles) is that "the highest sustained FP64 performance often comes from matrix-multiply kernels," and that codes not dominated by matrix multiplication are limited by memory bandwidth anyway, so more native FP64 wouldn't help them.

Emulation means using the low-precision hardware there's now plenty of to get results equivalent to native FP64. The method everyone points to is the Ozaki scheme, from 2012.

AMD is going the other way: the MI430X (in ORNL's Discovery, 2028) will have substantially more native FP64 than the MI355's 77 TFLOP/s. AMD hasn't given a number; HPCwire's guess from keeping the flop-to-byte ratio constant is around 190–200 TFLOP/s.

## Various perspectives

People who care a lot about this don't agree on what emulation is for. A few voices from the past year:

**Satoshi Matsuoka (RIKEN).** His June 2026 arXiv paper is titled "FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail." He argues Ozaki-II on FP8 Tensor Cores can replace hardware FP64, and that on GPUs like the B300 where the FP64 compute roof has collapsed, it pays off even for memory-bound kernels (GEMV, stencils, SpMV), not just GEMM. He also suggests AI coding assistants could make porting codes to emulation practical. (Eadline, HPCwire, Jun 2026.)

**Dan Stanzione (TACC).** TACC's new Horizon system (Blackwell) will run both native FP64 and Ozaki-emulated FP64, using NVIDIA's reference implementation, and verify every acceptance app both ways. He expects 2–3× over native FP64, and TACC will submit HPL to the TOP500 both ways. The full quote: "If there's a case that this emulation is producing scientifically acceptable results, then suddenly our FP64 cost is now divided by 2X to 3X without actually changing the chip." He also predicts that eventually emulation that "hit[s] all the corner conditions for IEEE 64" will just be hidden in microcode "and nobody knows anymore." (Prickett Morgan, The Next Platform, Feb 2026.)

**Darío Gil (DOE Under Secretary for Science).** On the Genesis Mission: "In discussions I've had with both [AMD CEO] Lisa Su and with [Nvidia CEO] Jensen [Huang], they have expressed a strong commitment for FP64, that it will continue. For us, it's very important, because we don't view this [as a] substitution. These are complementary." His point is that AI for science depends on high-precision simulation to generate training data. Note: Matsuoka's paper says Genesis "explicitly identified Ozaki emulation as its fallback path," and HPCwire itself pushed back on that: Gil never said it. (Woodie, HPCwire, Feb 2026; Eadline, HPCwire, Jun 2026.)

**Joseph George (AMD).** "If you get to wrong answers really fast, that's not helping anybody. We want to get to right answers fast." AMD supports Ozaki in software, but he hasn't seen definitive evidence it matches FP64 results, and expects a mixed-precision future that's "going to be workload-specific." (Woodie, HPCwire, Jun 2026.)

Also worth having in your back pocket, **Nick Malaya (AMD)**, the sharpest critique: Ozaki isn't IEEE-compliant and doesn't give the same answers as FP64; matrices whose elements "differ by a few orders of magnitude" cause accuracy problems; non-square matrices drop below native FP64 performance; and fewer than 10% of HPC applications are built around DGEMM, so "it doesn't address 90% of the HPC apps." "You need those FP64 pipes to fall back onto." (Woodie, HPCwire, Mar 2026.) Several of these points come back later in the talk: the magnitude-range trade-off (Ozaki-I), the dependence on matrix size (Ootomo, Brower), and ADP's fallback to native DGEMM.

Goal of the talk: trace the development of the Ozaki scheme, and consider what each work tells us about when it can or should be used.
