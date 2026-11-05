# Intro {background-color="#eef4fc"}

## FP64 on new GPUs

::: {style="font-size: 0.6em"}
| GPU | Year | FP64 Vector (TFLOP/s) | FP64 Tensor Core (TFLOP/s) |
|:----|:----:|----:|----:|
| V100 | 2017 | 7.8 | — |
| A100 | 2020 | 9.7 | 19.5 |
| H100 | 2022 | 34 | 67 |
| B200 | 2024 | 37 | 37 |
| **B300** | **2025** | **1.3** | **1.3** |
| Rubin | 2026 | 33 | 200 (emulated) |
:::

["The floating-point capability of the platform is not improved – not improved – over the previous generation." — Jack Dongarra, SC25 [@woodie2026genesis]]{style="font-size: 0.75em"}

. . .

**Emulation:** Use the abundant low-precision hardware to get results equivalent to native FP64. The method is the Ozaki scheme [@ozaki2012].

## Various perspectives

People who care a lot about FP64 don't agree on what emulation is for.

::: {style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.7em 1.4em; margin-top: 0.4em; font-size: 0.6em"}
::: {.fragment style="border-left: 4px solid #1c4e8a; padding: 0.2em 0.7em"}
"FP8 is All You Need: Debunking Hardware FP64 as the HPC Holy Grail"

[Satoshi Matsuoka, RIKEN [@eadline2026matsuoka]]{style="color: #666"}
:::

::: {.fragment style="border-left: 4px solid #1c4e8a; padding: 0.2em 0.7em"}
"We don't view this [as a] substitution. These are complementary."

[Darío Gil, U.S. Department of Energy [@woodie2026genesis]]{style="color: #666"}
:::

::: {.fragment style="border-left: 4px solid #1c4e8a; padding: 0.2em 0.7em"}
"If there's a case that this emulation is producing scientifically acceptable results, then suddenly our FP64 cost is now divided by 2X to 3X without actually changing the chip."

[Dan Stanzione, TACC [@morgan2026tacc]]{style="color: #666"}
:::

::: {.fragment style="border-left: 4px solid #1c4e8a; padding: 0.2em 0.7em"}
"If you get to wrong answers really fast, that's not helping anybody. We want to get to right answers fast."

[Joseph George, AMD [@woodie2026fp8]]{style="color: #666"}
:::
:::

. . .

**Goal:** Trace the development of the Ozaki scheme, and consider what each work tells us about when it can or should be used.
