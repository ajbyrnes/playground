# Intro

- Why talk about the Ozaki scheme?
- HPC measures itself using the TOP500 list, which ranks supercomputers based on their performance on double-precision LINPACK
- In other words, throughput of double-precision floating-point operations is the primary measure of performance for supercomputers
	- (Yes, there have been critiques of this since the beginning, but the TOP500 is what comes up if you look for "list of the fastest supercomputers")

- We are now in the exascale era of HPC
- For the last decade, we have increasingly seen GPUs being incorporated to reach exascale FLOPs
- Vendors are shifting their hardware designs to support AI workloads
- What does that mean? --> More hardware for low-precision floating-point and integer operations, less hardware for 32-bit and 64-bit floating-point operations
- AI workloads can be computed using low-precision units --> Results arrive faster and more cheaply
- Canonically, scientific applications (HPC workloads) need *high-precision units*, like FP32 or FP64

- Example: NVIDIA FP64 throughput shrinkage

- We may be able to squeeze more high-precision compute out of AI-oriented hardware via **emulation**
- **Emulation** = Use different algorithms for performing math, so that a problem on FP64 inputs can be computed using low-precision hardware, and then the solution has FP64-level accuracy

# Foundations

- Basic floating-point stuff
- We're going to be concerned with FP64 so just define everything in terms of FP64
- "Floating-point notation" = $\text{mantissa} \times 2^\text{exponent}$
	- Simplification of IEEE 754
- IEEE 754 binary64 --> 53-bit mantissa (in terms of precision), 11-bit exponent

- Mantissa space
- Floating-point space = Floating-point numbers visualized on a real number line --> Density decreases as exponent grows
- Mantissa space = Floating-point mantissas visualized as a row of bits, positioned relative to their magnitudes
- Mantissa space is important to the Ozaki scheme!
- Not exactly a standard technical term, but it will be very useful for us in this presentation
	- (Also some of the papers do kinda sorta refer to "mantissa space" so)
- Mantissa space is defined over a set of floating point numbers
- Width of mantissa space = Distance between the most significant bit of the highest magnitude value in the set and the least significant bit of the lowest magnitude value in the set
- Magnitude range = Difference in exponents of the highest magnitude values and the lowest magnitude values
- So magnitude range and width of mantissa space are closely correlated, but magnitude range could be measured in, like, binary or decimal, whereas width of mantissa space is always in terms of bits

- Rounding error in floating-point operations
- IEEE 754 says that the result of a floating-point operation should be as if it was performed "exactly" (i.e. to infinite precision) and then rounded to fit in the target format
- Rounding error of a single operation is bounded to (specific bounds for IEEE 754 binary64)
- Rounding error can accumulate over *sequences* of floating-point operations!
- Estimate max rounding error for a dot product and a matrix product
- There have been many different methods for attempting to reduce error:
	- Multi-word precision i.e. double-double or quad-double
	- Correcting with residuals i.e. compensated sums

# Ozaki-I

- The Ozaki scheme is a method for reducing the rounding error in a matrix multiplication

- Splitting

- The Ozaki scheme -- the actual procedure(s) for computing the matrix product
- First: Decompose inputs into slices
	- Option 1: Fully decompose --> Every slice product is error-free
	- Option 2: Partially decompose --> High-magnitude slice products are error-free, rounding error is only accumulated in the lower-order terms
- Second: Compute slice and residual products
- Third: Accumulate slice and residual products
	- Option 1: Standard FP64 accumulation --> Rounding error occurs in the additions only
	- Option 2: Accurate summation technique --> Little-to-no rounding error in the additions
- Result: Matrix product that is more accurate than standard FP64

- Performance
- Ozaki scheme works by decomposing one matrix product with rounding error into many matrix products without rounding error
- The decomposition (splitting procedure) is defined so that *standard matrix multiplication* will be error-free
- Therefore, the Ozaki scheme can be more efficient than other methods because it can use optimized BLAS routines

- Evaluation
- The evaluation of the Ozaki scheme in this paper is focused on demonstrating that it is a more-efficient alternative to *other accurate matrix product methods*
- Hardware: Intel Core 2 Duo, Intel Core 2 Extreme
- Software: MATLAB
- The derivation and evaluation of the Ozaki scheme does note a performance and accuracy trade-off which is controlled by **magnitude range**
- The test cases are on narrow, synthetic inputs -- but this is fine for the context of the paper

## Discussion

- Summary of performance story

- Summary of evaluation evidence

- Verdict re: can we replace FP64

- Future work

## Conclusion: Everything else

- NVIDIA is not the only GPU manufacturer
- AMD's data-center/HPC class chips have had better FP64 throughput than NVIDIA's for a few years
- So, like, maybe all this means is that we use AMD hardware instead of NVIDIA hardware
- The concern there would still be, like, national labs want to host their own AI training and inference, so like you might need dedicated hardware for that still
- Basically that we actually split into groups of "general PC hardware that doesn't have particularly high throughput for anything", "specialized low-precision throughput hardware for AI" and then "specialized high-precision throughput hardware for HPC"

- Is INT8 reliable?
- INT8 is maybe not as useful for AI as we initially thought
- Manufacturers may be decreasing INT8 support in favor of FP8 support
- In which case, Ozaki-I runs into the overhead problems... again...

- Ozaki-II
- But the INT8 issue may not even matter, because there is the Ozaki-II scheme