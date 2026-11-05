# Ozaki in Applications {background-color="#eef4fc"}

[Early Examples · DMRG · Results]{style="color: #666"}

## Early examples

[**Early Examples** → DMRG → Results]{style="font-size: 0.7em; color: #666"}

Until this point, every evaluation has been on synthetic workloads. Are there matrix products in actual applications that would benefit from emulation?

::::: columns
::: {.column .fragment width="50%"}
**Quantum circuit simulation** [@ootomo2024]

2–4.85× faster than cuBLAS ZGEMM, same or better accuracy

[(Random unitary circuits: "toy models")]{style="font-size: 0.7em; color: #666"}
:::

::: {.column .fragment width="50%"}
**QR factorization** [@schwarz2026]

Without ADP: 5× faster, error 2.9e-12

With ADP: 3.5× faster, error 7e-16
:::
:::::

::: fragment
**Neither is a full, end-to-end application workload.**
:::

## Brower et al. (2026)

[Early Examples → **DMRG** → Results]{style="font-size: 0.7em; color: #666"}

*Mixed-Precision Ab Initio Tensor Network State Methods Adapted for NVIDIA Blackwell Technology via Emulated FP64 Arithmetic*

::::: columns
::: {.column .fragment width="60%"}
[Cole Brower, Samuel Rodriguez Bernabeu, Jeff Hammond, John Gunnels [NVIDIA]{style="color: #666"}\
Sotiris S. Xantheas [Pacific Northwest National Laboratory; University of Washington]{style="color: #666"}\
Martin Ganahl [SandboxAQ]{style="color: #666"}\
Andor Menczer, Örs Legeza [Wigner Research Centre for Physics]{style="color: #666"}]{style="font-size: 0.6em"}

Published in *Journal of Chemical Theory and Computation*, 2026 [@brower2026]
:::

::: {.column .fragment width="40%"}
::: {style="border: 2px solid #1c4e8a; border-radius: 8px; padding: 0.8em 1em; margin-top: 1.2em; background: #eef4fc"}
**Ozaki emulation in an actual scientific application**
:::
:::
:::::

## DMRG

[Early Examples → **DMRG** → Results]{style="font-size: 0.7em; color: #666"}

**DMRG:** Iterative algorithm for finding the lowest-energy state of a molecule

. . .

Every iteration: Many DGEMMs, some in rounding-sensitive steps (eigensolver, SVD)

. . .

**Bond dimension $D$:** Higher is more accurate, but takes more memory and compute

## From $D$ to DGEMMs

[Early Examples → **DMRG** → Results]{style="font-size: 0.7em; color: #666"}

The matrices in DMRG are $D \times D$. Symmetries split them into smaller blocks, and the DGEMMs run on the blocks, not on the whole matrix.

[DGEMM structure from the group's hybrid CPU-GPU implementation [@menczer2025hybrid]]{style="font-size: 0.6em; color: #666"}

. . .

**A lot of DGEMM work, but it arrives as many DGEMMs much smaller than $D$.**

## Accuracy

[Early Examples → DMRG → **Results**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="50%"}
![](../figures/papers/brower/fig01.png){height="390px" fig-align="center"}

[**Small system:** 6 and 4 slices track FP64; 2 miss the target, and 3 is borderline.]{style="font-size: 0.65em"}
:::

::: {.column .fragment width="50%"}
![](../figures/papers/brower/fig03.png){height="390px" fig-align="center"}

[**Large system:** 4 and 6 slices agree with FP64 over 700 iterations.]{style="font-size: 0.65em"}
:::
:::::

## Performance

[Early Examples → DMRG → **Results**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="55%"}
![](../figures/papers/brower/fig04.png){height="400px"}
:::

::: {.column width="45%"}
::: fragment
**Eager:** Emulate every DGEMM. Slower than native FP64, even with 2 slices.
:::

<br>

::: fragment
**Performant:** The library decides. Tracks native FP64: It appears to always choose *not* to emulate.
:::

<br>

::: fragment
**The fallback works, but that's a bad sign for "emulation speeds up workloads."**
:::
:::
:::::

## Performance at scale

[Early Examples → DMRG → **Results**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="62%"}
![](../figures/papers/brower/fig05.png){height="440px"}
:::

::: {.column width="38%"}
::: {style="font-size: 0.85em"}
Native FP64 vs. performant emulation on B200, with H100 for reference

::: fragment
**Performant mode tracks native FP64:** Even at scale, it appears to choose native DGEMM.
:::
:::
:::
:::::

## What the application tells us

[Early Examples → DMRG → **Results**]{style="font-size: 0.7em; color: #666"}

::: fragment
**Accuracy:** Four to six slices reach chemical accuracy. Two slices break DMRG entirely.
:::

<br>

::: fragment
**Speed:** No real gain yet. Emulation is "often slower for memory-bound matrix multiplications", and sector DGEMMs are small.
:::

<br>

::: fragment
**Caveat:** Prerelease cuBLAS, and the paper defers performance to a later study.
:::

## Takeaways: Brower et al. (2026)

::: {style="border: 2px solid #1c4e8a; border-radius: 8px; padding: 0.8em 1.2em; margin-top: 1.2em; background: #eef4fc"}
**Role:** FP64 emulation inside a real scientific code, with the library deciding DGEMM by DGEMM

. . .

**Trade-off:** Accuracy holds with few slices, but speedup depends on DGEMM size, which is set by sector sizes, not $D$
:::
