# Ozaki on IMMU

## Ozaki is shaped like Tensor Cores

In 2018, NVIDIA introduced Tensor Cores, which are dedicated mixed-precision matrix-multiplication circuits. Inputs can be in lower precisions than outputs.

In the FP64 version of the Ozaki scheme, the slices are deliberately restricted so that only a fraction of the mantissa bits in an FP64 value are filled. That headroom exists for the result, but there's no reason the slices themselves can't be represented using smaller formats. Functionally, the inputs in our previous example are 25-bits and the result is accumulated in FP64.

## Integer Slices

The biggest hurdle to using INT8 Tensor Cores is in setting the INT8 representation of the slices. 
Ootomo et al. settle on a block-float representation with an implicit exponent and separate vectors carrying explicit exponent information.
First, they extract the exponent information by calculating a row-wise or column-wise scaling factor for each matrix. Then they remove the exponent information by dividing the matrices by the scaling vectors. 
Now, the mantissa space has been rescaled so that the most significant bit of the largest magnitude value starts at $2^{-1}$. Slices can then be represented in block-float format, where each slice is a block of integers with the shared exponent. The shared exponent is determined by the index of the slice.

## The IMMU Ozaki Scheme
Step 1: Decompose into slices and scaling vectors
Step 2: Perform slice products on integer cores, then use implicit shared exponents and explicit scaling factors to scale the result back to FP64
Step 3: Accumulate the FP64 slice products

## What the Slices Leave Out

The paper never says outright what happens to the mantissa bits that don't fit in the slices, but everything points to them being cut off. With a fixed number of INT8 slices, the kept mantissa space is "the number of splits × BPS", and anything below that is lost. ozIMMU "uses bit operations to cut the mantissa" to make the slices. INT8-AUTO picks the slice count so that the "average mantissa loss in the splitting process" stays under a threshold $T$, and $T = 0$ means no mantissa loss. Very small elements are "underflowed and treated as zeros when splitting the matrix."

Not all slice products are computed either. It suffices to sum $A^{(i)} B^{(j)}$ for $i + j \le s + 1$, "since the other results only have a negligible effect on the resulting accuracy." That's $s(s+1)/2$ products instead of $s^2$: 45 instead of 81 for INT8x9.

So INT8xX approximates $AB$ in two ways: bits below the last slice are dropped, and low-order slice products are skipped. In the authors' words, "the main error comes from the insufficient splitting and not the rounding error of the accumulation." This is why INT8x9 degrades as $\phi$ grows: a wider mantissa space leaves more bits below the 9th slice.

# Evaluation

## Experimental Setting
Inputs: Same synthetic matrices used by Ozaki et al. (2012), $\phi = 0.1, 1, 2, 4$
Hardware: A100, TITAN RTX, RTX A6000, RTX 6000 Ada.
Error measurement: Maximum componentwise relative error, computed against a double-double reference.
Variants: INT8x9 (9 slices), INT8x10 (10 slices), INT8x11 (11 slices)

## Results

Accuracy: At low $\phi$, all variants maintain relative error of $10^{-16}$. As $\phi$ increases, accuracy degrades. Lower-slice variants see accuracy degrade earlier

Throughput: A100 (1:32), TITAN RTX (1:522), RTX A6000 (1:515), RTX 6000 Ada (1:520). Ozaki throughput on A100 is limited to 8 TFlop/s at best, well below DGEMM's peak of ~17 TFlop/s. On TITAN RTX, Ozaki throughput is up 3 TFlop/s over DGEMM's 0.5 TFlop/s. On RTX A6000, Ozaki throughput is up to 4 TFlop/s over DGEMM's 0.5 TFlop/s. On RTX 6000 Ada, Ozaki throughput is up to 6 TFlop/s over DGEMM's 1 TFlop/s. Ozaki throughput grows with problem size.  
  
Power efficiency: Power efficiency (GFlops/W) tracks with throughput. It's worse than DGEMM on the A100 and better than DGEMM on TITAN RTX, RTX A6000, and RTX 6000 Ada.  
  
Average power consumption: On A100, Ozaki power consumption increases with problem size until reaching the same level as DGEMM, which is right at TDP. On TITAN RTX, RTX A6000, and RTX 6000 Ada, DGEMM remains at about 1/2 TDP while Ozaki reaches TDP (Except for the smallest problem sizes).
  
Time breakdown of INT8x9: Basicaslly the same on A100 and RTX6000 Ada. In general, time is dominated by FP64 accumulation and INT8 GEMMs. 

# Discussion

## Purpose of the Ozaki scheme/What the evaluations tell us

On some hardware, the Ozaki scheme can be used not just to produce a more accurate matrix product but to produce a matrix product *faster* than the standard DGEMM routine. In these cases, it is also more power efficient than running DGEMM. 

Testing is still limited to synthetic matrices, within a very narrow range of sizes. We can see indications that the Ozaki scheme only provides speedup for *large enough* matrices, but we don't see this carried all the way to the extremes.  
  
Actual memory overhead is not measured.

## Implementation analysis

Representing slices as INT8 means that the number of bits per slice is limited to 8 at most. The result is that we need more slices to fully decompose a matrix. However, because INT8 is a smaller format than FP64, a single FP64 slice is equivalent to (blank) INT8 slices. So we can often come out ahead on memory usage even though we need more slices.  
  
The same thing happens with compute. On hardware where INT8 throughput is sufficiently greater than FP64 throughput, we are able to chew through all the slices faster than doing a single FP64 matrix multiplication.  
  
Even though the evaluation results open up a whole new use-case for the Ozaki scheme, the implementation presents challenges to actually using it this way. Users have to manually tune slice count based on their desired accuracy, which can be difficult to know, the magnitude range of their matrices (admittedly easy to learn but still can be computationally expensive at small sizes), and the hardware they're using (i.e. how many slices are equivalent to one FP64 slice). 


