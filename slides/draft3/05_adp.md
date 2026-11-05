# Ozaki with ADP {background-color="#eef4fc"}

[FP64 Emulation · ESC · ADP]{style="color: #666"}

## Schwarz et al. (2026)

[**FP64 Emulation** → ESC → ADP]{style="font-size: 0.7em; color: #666"}

*Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme*

[Angelika Schwarz, Anton Anders, Cole Brower, Harun Bayraktar, John Gunnels, Kate Clark, RuQing G. Xu, Samuel Rodriguez, Sebastien Cayrols, Pawel Tabaszewski, Victor Podlozhnyuk [NVIDIA]{style="color: #666"} · *SCA/HPCAsia '26*, 2026 [@schwarz2026]]{style="font-size: 0.55em"}

. . .

2025: FP64 emulation added to cuBLAS [@nvidia2025emulation]

. . .

::: {style="border: 2px solid #1c4e8a; border-radius: 8px; padding: 0.8em 1em; margin: 0.6em auto 0; background: #eef4fc; width: 70%; text-align: center"}
**Use emulation when it is beneficial, and standard DGEMM when it is not**
:::

## A new goal

[**FP64 Emulation** → ESC → ADP]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="50%"}
**Ozaki et al. (2012)**

Goal: As little rounding error as possible

Slice width chosen so slice products are provably exact
:::

::: {.column .fragment width="50%"}
**Schwarz et al. (2026)**

Goal: FP64-equivalent accuracy

Slices always contain eight significant bits
:::
:::::

<br>

::: fragment
**Aiming for FP64-equivalent accuracy gives us a foothold for reasoning about how many slices we need.**
:::

## Every slice uses all 8 bits

[FP64 Emulation → **ESC** → ADP]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/eight_bit_slices.svg){height="330px" fig-align="center"}

Slices are cut from the largest value down, eight bits at a time.

::: fragment
**How many slices do we actually need?**
:::

## Which bits matter

[FP64 Emulation → **ESC** → ADP]{style="font-size: 0.7em; color: #666"}

::: r-stack
![](../figures/static/slides/esc_matrix_1.svg){height="290px"}

![](../figures/static/slides/esc_matrix_2.svg){.fragment height="290px"}

![](../figures/static/slides/esc_matrix_3.svg){.fragment height="290px"}
:::

Slices should cover every mantissa bit of the largest values, and of the values behind the largest term.

::: fragment
**ESC:** $0 + 0 - (-11) = 11$ extra bits
:::

## Slices over 53 + ESC bits

[FP64 Emulation → **ESC** → ADP]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/esc_slices.svg){height="285px" fig-align="center"}

Slices cover $53 + \text{ESC}$ bits. The rest is dumped.

::: fragment
**Enough for FP64-equivalent accuracy? Asserted, not proven.**
:::

## Automatic Dynamic Precision (ADP)

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="45%"}
![](../figures/papers/schwarz/fig08.png){height="470px"}
:::

::: {.column width="55%"}
::: {style="font-size: 0.8em"}
::: fragment
Check for Inf/NaN, compute ESC, then decide: Emulate, or fall back to native DGEMM
:::

<br>

::: fragment
**Goal:** A matrix product at least as accurate as native FP64, in at least the same amount of time
:::

<br>

::: fragment
How the heuristic decides "emulation slower than FP64?" is not described.
:::
:::
:::
:::::

## Experimental setting

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

- **Inputs:** Demmel et al.'s test matrices, with a parameter $b$ controlling exponent range [@demmel2026grade]
- **Hardware:** GB200, RTX Pro 6000 Blackwell Server Edition
- **Accuracy:** Maximum componentwise relative error, against a standard FP64 matrix multiplication
- **Variants:** With and without fallback to native DGEMM

## Evaluation: Accuracy

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

![](../figures/papers/schwarz/fig02.png){height="440px" fig-align="center"}

. . .

**With fallback:** Accuracy indistinguishable from native DGEMM.

## Evaluation: Speed

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

![](../figures/papers/schwarz/fig06.png){height="510px" fig-align="center"}

[FP64 : INT8 peak: **GB200** 1 : 125 · **RTX Pro 6000** 1 : 512]{style="font-size: 0.5em; color: #666"}

. . .

**With fallback:** Up to 2.3× faster on GB200, and 13.2× on RTX Pro 6000.

## What the evaluations tell us

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

The Ozaki scheme is now an explicit alternative to DGEMM, but it is still evaluated the way it was when it was an accurate matrix multiplication scheme.

. . .

**But:** No adversarial inputs, no look inside the fallback heuristic, and memory overhead is still not measured.

## Impacts of implementation

[FP64 Emulation → ESC → **ADP**]{style="font-size: 0.7em; color: #666"}

::: fragment
**Accuracy:** Slice products are no longer guaranteed exact, though this only matters for very large matrices
:::

<br>

::: fragment
**Memory:** Eight significant bits in every slice is the best memory overhead the scheme can get
:::

<br>

::: fragment
**Usability:** Emulation is invisible to the user... but the decision to emulate is a black box
:::

## Takeaways: Schwarz et al. (2026)

::: {style="border: 2px solid #1c4e8a; border-radius: 8px; padding: 0.8em 1.2em; margin-top: 1.2em; background: #eef4fc"}
**Role:** FP64 emulation, a drop-in replacement for DGEMM used only when it is beneficial

. . .

**Trade-off:** Handled automatically by ESC and ADP, but how ADP makes that call is not described
:::
