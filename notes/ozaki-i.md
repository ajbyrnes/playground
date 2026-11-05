# Key idea: Splitting to perform exact products

## Splitting

Rump et al. propose a splitting mechanism for splitting a single value $x$ into a high part $x^{(1)}$ and a low part $\underline{x}^{(2)}$.
The splitting mechanism uses a splitting factor $\sigma$, which is a power of 2, parameterized by $M$.
The splitting mechanism has the property that $x^{(1)}$ has at most $53-M$ significant bits.

## Decomposition of a product
The elements in a matrix product with inner dimension $n$ are dot-products between vectors of length $n$.
The dot-product of two length-$n$ vectors involves $n$ multiplications and $n - 1$ additions, where floating-point error can accumulate.

We can apply the splitting procedure to a vector! In this case, we scale the splitting factor $\sigma$ to the maximum magnitude element in the vector.
Splitting can be applied iteratively to each remainder $\underline{x}^{(i)}$ so that the vector $x$ is expanded into a sum of slices, each with at most $53-M$ significant bits.
We could do the same with a vector $y$.
Then we can represent the dot product $x^T y$ as a sum of slice products: $(x^{(1)} + \cdots + x^{(n_x)})(y^{(1)} + \cdots + y^{(n_y)})$, where each slice product $x^{(i)}y^{(j)}$ is *exact*.
Then rounding error only accumulates over the $n - 1$ additions. --> $x^T y$ is more exact than if it had been computed directly!

## The Ozaki Scheme
Step 1: Decompose into slices, fully or partially
Step 2: Perform slice products. Slices are standard FP64 matrices, and can be multiplied with standard FP64 matrix multiplication routines
Step 3: Accumulate, either accepting rounding error or with some accurate summation technique

## Tradeoff: Accuracy vs performance
Magnitude range controls a trade-off between accuracy and performance.
The idea in creating the slices is that the slices should cover the whole mantissa space. If some part of mantissa space is not covered by a slice, then it is represented inexactly.
Wider mantissa space requires more slices to be covered.
More slices means more slice products --> More memory overhead and more computation.

# Evaluation

## Experimental Setting

Intel Core 2 Duo, Intel Core 2 Extreme
Synthetic matrices: Mantissas drawn from uniform distribution, exponents drawn from normal distribution and multiplied by $\phi$, which controls magnitude range
Matrix sizes are $n = 1000$ and $n = 2000$
Implemented via MATLAB

## Results

The Ozaki scheme with partial decomposition (2 slices, 3 slices, 4 slices) is compared against standard FP64 multiplication in MATLAB, the lssresidual procedure from INTLAB, and XBLAS.
Results show that:
1. Ozaki scheme is, generally, much faster than XBLAS by a factor of (blank) and more accurate than standard multiplication and INTLAB by a factor of (low range - high range).
2. Ozaki scheme accuracy/speed is sensitive to magnitude range $\phi$. The 4 slice scheme takes longer than the 3 slice scheme which takes longer than the 2 slice scheme. The schemes have comparable accuracy at low $\phi$, but as $\phi$ increases, the schemes with lower slice counts see more accuracy degradation.

# Discussion

## Purpose of the Ozaki scheme/What the evaluations tell us

At this point, the Ozaki scheme is only meant to be an efficient and more accurate way of performing matrix multiplication.
The evaluations are built around this. The testing in this work is done on small problems and consumer-grade hardware. 
The evaluations do generally prove that the Ozaki scheme is able to be more efficient than other methods for performing accurate matrix multiplication, though not over a wide range of cases.
The work also identifies the accuracy-vs-performance trade-off that is driven by magnitude range, but does not test this in the extreme.

## Implementation analysis
The splitting scheme is designed to produce regular floating-point matrices so that slice products can be computed via optimized BLAS. This is a clever way to essentially transform a problem so that it can exploit existing efficient routines. Ozaki is really a wrapper *around* matrix multiplication, which also makes it promising as a method that could be integrated transparently into existing libraries.

Memory overhead is a huge problem. A single split means double the memory requirements for a single matrix multiplication. Two splits triples it, and so on.  
  
This is still tricky for users, but maybe not worth bringing up here. But users are left to manage the accuracy-vs-speedup tradeoff themselves. The evaluation shows the general relationship that $\phi$ has with this trade-off, but not really enough that users could just pick a value of $k$ based on their matrices $\phi$.
 