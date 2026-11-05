# Ozaki on IMMU {background-color="#eef4fc"}

[Tensor Cores · Integer Slices · New Possibilities]{style="color: #666"}

## Ozaki is shaped like Tensor Cores

[**Tensor Cores** → Integer Slices → New Possibilities]{style="font-size: 0.7em; color: #666"}

**Tensor Cores (2018):** Dedicated mixed-precision matrix multiplication circuits. Inputs can be in lower precision than outputs.

. . .

::::: columns
::: {.column width="55%"}
![](../figures/static/slides/split_scalar_d3.svg){width="100%"}
:::

::: {.column width="45%"}
[FP64 slices are deliberately restricted to a fraction of the mantissa (27 of 53 bits here). That headroom exists only so the *result* can be exact in FP64.]{style="font-size: 0.8em"}
:::
:::::

. . .

**There's no reason the slices themselves can't use smaller formats.**

[The first Ozaki scheme on Tensor Cores (2020) used FP16-FP32 cores [@mukunoki2020]]{style="font-size: 0.7em; color: #666"}

## Ootomo et al. (2024)

[**Tensor Cores** → Integer Slices → New Possibilities]{style="font-size: 0.7em; color: #666"}

*DGEMM on integer matrix multiplication unit*

[Hiroyuki Ootomo, Katsuhisa Ozaki, Rio Yokota · *The International Journal of High Performance Computing Applications*, 2024 [@ootomo2024]]{style="font-size: 0.6em; color: #666"}

. . .

::: {style="border: 2px solid #001E61; border-radius: 8px; padding: 0.8em 1em; margin: 0.8em auto 0; background: #eef4fc; width: 70%; text-align: center"}
**The Ozaki scheme on integer Tensor Cores (INT8)**
:::

## Integer slices

[Tensor Cores → **Integer Slices** → New Possibilities]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/immu_blockfloat.svg){width="100%"}

Ootomo et al. use a block-float format: Each slice contains integer values aligned to a shared exponent.

. . .

**Exponents are divided out and stored in separate vectors.**

## The IMMU Ozaki scheme

[Tensor Cores → **Integer Slices** → New Possibilities]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/immu_product_d3.svg){height="200px" fig-align="center"}

::: {.incremental style="font-size: 0.85em"}
1. **Decompose:** Split into INT8 slices and scaling vectors
2. **Multiply:** Compute slice products on integer Tensor Cores, then convert products back to FP64 using the shared exponents and scaling vectors
3. **Accumulate:** Sum the FP64 products
:::

## What the slices leave out

[Tensor Cores → **Integer Slices** → New Possibilities]{style="font-size: 0.7em; color: #666"}

With $s$ INT8 slices, the product is only approximated:

$$
AB \approx \sum_{i + j \,\le\, s + 1} A^{(i)} B^{(j)}
$$

Bits below the last slice are cut off, and products with $i + j > s + 1$ are skipped.

. . .

**The main error comes from insufficient splitting, not from rounding.**

## Experimental setting

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

- **Inputs:** Same synthetic matrices as Ozaki et al. (2012), $\phi = 0.1, 1, 2, 4$
- **Hardware:** A100, TITAN RTX, RTX A6000, RTX 6000 Ada
- **Error:** Maximum componentwise relative error, against a double-double reference
- **Variants:** INT8x9, INT8x11, INT8x13 (9, 11, 13 slices)

## Results: Accuracy

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="46%"}
![](../figures/papers/ootomo/fig07.png){width="100%"}
:::

::: {.column width="54%"}
At low $\phi$, all variants stay near $10^{-16}$. As $\phi$ grows, variants with fewer slices degrade first.

::: fragment
**We see again that lower slice counts improve performance at the cost of accuracy.**
:::
:::
:::::

## Results: Throughput

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="66%"}
![](../figures/papers/ootomo/fig09_labeled.png){width="100%"}

::: {style="font-size: 0.5em; line-height: 1.3; color: #666"}
FP64 : INT8 peak: **A100** 1 : 32 · **TITAN RTX** 1 : 522 · **RTX A6000** 1 : 515 · **RTX 6000 Ada** 1 : 520
:::
:::

::: {.column width="34%"}
::: {style="font-size: 0.75em"}
::: fragment
**A100:** At best 8 TFlop/s, vs. DGEMM's ~17.

**Consumer GPUs:** Up to 3, 4, 6 TFlop/s, vs. DGEMM's 0.5–1.
:::

::: fragment
**Speedup depends on the hardware. The new implementation let's us take proper advantage of what the hardware provides.**
:::
:::
:::
:::::

## New possibilities

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

On some hardware, the Ozaki scheme can be used not just to produce a more accurate matrix product, but to produce it *faster* than the standard DGEMM routine, and more power efficiently.

. . .

**But:** Testing is still limited to synthetic matrices, in a narrow range of sizes. Speedup only shows up for *large enough* matrices, and memory overhead is not measured.

## More slices, smaller slices

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

::: fragment
**Memory:** INT8 slices hold at most 8 bits, so we need more of them. But one FP64 slice takes the memory of 8 INT8 slices, so we can often come out ahead.
:::

<br>

::: fragment
**Compute:** Where INT8 throughput is high enough, we can compute some number of INT8 slice products faster than one FP64 matrix multiplication.
:::

## Usability

[Tensor Cores → Integer Slices → **New Possibilities**]{style="font-size: 0.7em; color: #666"}

Users have to manually tune the slice count, based on:

::: incremental
- **Desired accuracy:** Can be difficult to know
- **Magnitude range of their matrices:** Easy to measure, but costly at small sizes
- **Their hardware:** How many slices are equivalent to one FP64 matrix multiplication
:::

## Takeaways: Ootomo et al. (2024)

::: {style="border: 2px solid #001E61; border-radius: 8px; padding: 0.8em 1.2em; margin-top: 1.2em; background: #eef4fc"}
**Role:** Faster than the standard DGEMM routine, on some hardware

. . .

**Trade-off:** Unchanged, but with more (smaller) slices, and users still tune the slice count manually
:::
