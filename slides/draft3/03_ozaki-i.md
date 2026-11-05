# The Ozaki Scheme {background-color="#eef4fc"}

[Method · Trade-off · Purpose]{style="color: #666"}

## Rounding in mantissa space

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/addition_rounding_d3.svg){height="345px" fig-align="center"}

. . .

The result is rounded only if it doesn't fit in the 53-bit mantissa.

. . .

If the operands have few enough significant bits, the result is exact.

## Splitting

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

Rump et al. split a single value $x$ into a high part $x^{(1)}$ and a low part $\underline{x}^{(2)}$ [@rump2008a].

$$
x^{(1)} = \text{fl}\big((\sigma + x) - \sigma\big) \qquad \underline{x}^{(2)} = \text{fl}(x - x^{(1)})
$$

::::: columns
::: {.column width="56%"}
![](../figures/static/slides/split_scalar_d3.svg){width="100%"}
:::

::: {.column width="44%"}
::: {style="font-size: 0.8em"}
::: fragment
**Splitting factor:** $\sigma$, a power of 2, parameterized by $M$
:::

::: fragment
**Property:** $x^{(1)}$ has at most $53 - M$ significant bits
:::
:::
:::
:::::

## Decomposition of a product

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

![](../figures/static/slides/matmul_dims_d3.svg){height="215px" fig-align="center"}

The elements of a matrix product with inner dimension $n$ are dot products between vectors of length $n$.

. . .

Each dot product has $n$ multiplications and $n - 1$ additions, where floating-point error can accumulate.

## Splitting a vector

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="62%"}
![](../figures/static/slides/slice_vector_bits_d3.svg){height="340px"}
:::

::: {.column width="38%"}
![](../figures/static/slides/slice_vector_slices.svg){width="100%"}
:::
:::::

Scale $\sigma$ to the largest element of $x$, then split each remainder again.

. . .

**Result:** $x$ is a sum of slices, each with at most $53 - M$ significant bits

## Exact slice products

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

Split $y$ the same way, and the dot product becomes a sum of slice products:

$$
x^T y = \big(x^{(1)} + \cdots + x^{(n_x)}\big)^T \big(y^{(1)} + \cdots + y^{(n_y)}\big) = \sum_{i=1}^{n_x} \sum_{j=1}^{n_y} \underbrace{{x^{(i)}}^T y^{(j)}}_{\text{exact}}
$$

. . .

Rounding error only accumulates when the slice products are added together.

. . .

**$x^T y$ is more accurate than if it had been computed directly.**

## The Ozaki scheme

[**Method** → Trade-off → Purpose]{style="font-size: 0.7em; color: #666"}

::: incremental
1. **Decompose:** Split $A$ and $B$ into slices, fully or partially
2. **Multiply:** Slices are standard FP64 matrices, so slice products use standard FP64 matrix multiplication routines
3. **Accumulate:** Accept rounding error in the sum, or use an accurate summation technique
:::

## Trade-off: Accuracy vs. performance

[Method → **Trade-off** → Purpose]{style="font-size: 0.7em; color: #666"}

::::: columns
::: {.column width="67%"}
![](../figures/static/slides/span_slices_d3.svg){width="100%"}
:::

::: {.column width="33%"}
::: {style="font-size: 0.75em"}
Slices should cover the whole mantissa space. Any part that isn't covered is represented inexactly.

::: fragment
**Wider mantissa space requires more slices to cover it, and more slices means more memory overhead and more computation.**
:::
:::
:::
:::::

## Test matrices

[Method → **Trade-off** → Purpose]{style="font-size: 0.7em; color: #666"}

Ozaki et al. generate test matrices with a controlled magnitude range:

$$
a_{ij} = \underbrace{(\text{rand} - 0.5)}_{\text{uniform mantissa}} \times \underbrace{e^{\,\phi \cdot \text{randn}}}_{\text{normal exponent}}
$$

. . .

$\phi$ is the parameter controlling magnitude range.

. . .

We'll keep seeing matrices like these in the later evaluations.

## The trade-off in the results

[Method → **Trade-off** → Purpose]{style="font-size: 0.7em; color: #666"}

[**M1** FP64 · **M2** `lssresidual` (INTLAB) · [**M3–M5** Ozaki scheme, 2, 3, 4 slices]{style="color: #D50032"} · **M6** XBLAS]{style="font-size: 0.6em"}

::::: columns
::: {.column width="56%"}
[**Accuracy** (maximum componentwise relative error)]{style="font-size: 0.55em"}

![](../figures/papers/ozaki/table03_accmul.png){width="100%"}
:::

::: {.column width="44%"}
[**Time** relative to standard FP64]{style="font-size: 0.55em"}

![](../figures/papers/ozaki/table04_accmul.png){width="100%"}
:::
:::::

. . .

[**As $\phi$ increases, the schemes with lower slice counts see more accuracy degradation. More slices take longer.**]{style="font-size: 0.85em"}

## What it was for

[Method → Trade-off → **Purpose**]{style="font-size: 0.7em; color: #666"}

At this point, the Ozaki scheme is only meant to be an efficient and more accurate way of performing matrix multiplication.

. . .

**Tested on:** Intel Core 2 Duo and Core 2 Extreme, MATLAB, $n = 1000$ and $2000$

**Result:** Accuracy near XBLAS at low $\phi$, in a much smaller fraction of the time (8–35× faster)

. . .

The evaluations show the Ozaki scheme can be more efficient than other methods for accurate matrix multiplication, though not over a wide range of cases.

## A wrapper around matrix multiplication

[Method → Trade-off → **Purpose**]{style="font-size: 0.7em; color: #666"}

Slices are regular FP64 matrices, so most of the work can be offloaded to optimized BLAS routines.

. . .

**The Ozaki scheme is essentially a wrapper around DGEMM, transforming one DGEMM call into several.**

. . .

This also makes it promising as a method that could be integrated transparently into existing libraries.

## Takeaways: Ozaki et al. (2012)

::: {style="border: 2px solid #001E61; border-radius: 8px; padding: 0.8em 1.2em; margin-top: 1.2em; background: #eef4fc"}
**Role:** A faster way of performing accurate matrix multiplication

. . .

**Trade-off:** Magnitude range determines the slice count, which trades accuracy against performance and memory
:::
