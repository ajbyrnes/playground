# Automatic Dynamic Precision

## Ozaki as FP64 emulation

This work builds on the implementation of ozIMMU, pursuing the result that shows Ozaki on INT8 Tensor Cores may be *faster* than standard double precision matrix multiplication on some hardware. 

Two ideas fall out of the change in thinking about the purpose of the Ozaki scheme. The one obvious one is the implementation of a system that can dynamically determine whether emulation will be more performant and at least as accurate. The less obvious idea is that the shift to FP64-equivalent accuracy, rather than correct-rounding accuracy, as a goal, gives us a foothold for reasoning about how many slices we need.

## Exponent Span Capacity

One change in this implementation is to have every slice contain eight significant bits, no matter what. (This is kinda an awkward side note that just needs to get established lol.) 
  
How many slices do we then need to achieve FP64-equivalent accuracy? The authors suppose that the slices should at least include all of the mantissa bits of the values that make the most significant contributions to the result. This means, of course, the highest magnitude values, but it also means the values that contribute to the highest magnitude element in the resulting matrix product. It is asserted without proof that if the slices cover these mantissa bits, then FP64 equivalent accuracy will be maintained.

Thus we need at least enough slices to cover the top 53 mantissa bits of mantissa space (the highest magnitude values), and then we need additional slices to cover down to the least significant bit of the values contributed to the maximum magnitude element in the result. The authors label the distance between the 53rd bit in mantissa space and the least significant bit of the relevant values "Exponent Span Capacity" or ESC.  
  
Clearly show relatinoship between ESC and slice counts.

### The example on the slides

ESC is defined one dot product at a time, so the figure looks at one entry of C, $c_{11}$: row 1 of A times column 1 of B. The largest value in the row is $a_{11}$ (exponent 0), and the largest in the column is $b_{31}$ (exponent 0). The three terms of $c_{11}$ are about $2^{-20}$, $2^{-11}$, and $2^{-30}$, so the largest term is $a_{12} b_{21}$, which comes from $a_{12}$ (exponent $-3$) and $b_{21}$ (exponent $-8$). ESC $= 0 + 0 - (-11) = 11$. The ESC for the whole product is the largest ESC over all the entries of C; in this example the other entries all have small ESC, so it's 11.

Only exponents matter here, mantissas are ignored. The paper adds 1 to the ESC as a safety margin, because multiplying two mantissas can bump the exponent up by one. The figures leave that out.

The slides also handwave the sign: every slice is drawn as 8 mantissa bits. In the real implementation the first slice also carries the sign (the rest are read as unsigned, and stored as signed int8 with a carry trick), so slice count is $\lceil (54 + \text{ESC}) / 8 \rceil$, not $(53 + \text{ESC}) / 8$. Not worth the time in the talk.

On the slicing slide: each row of A (and column of B) is lined up on its largest value, then slices of 8 bits are laid over $53 + \text{ESC} = 64$ bits, exactly 8 slices. $a_{12}$ is 3 bits down from the top and $b_{21}$ is 8 bits down, so all 53 of their mantissa bits land inside the slices. Bits past the last slice are dumped: here that's the tail of $a_{13}$ and $b_{11}$, which only show up in small terms.

## Automatic Dynamic Precision

Using ESC, the authors can develop a system that decides whether emulation is desirable. ESC is calculated for the pair of matrices to be multiplied, providing an upper bound on the number of slices needed to achieve a product at least equivalent to a double precision matrix multiplication. Then, a decision can be made about whether performing the Ozaki scheme with this number of slices would be effiicient on the given hardware.  If not, a regular double-precision matrix multiplicaction is performed.

This cleanly removes any issues users might face in deploying emulation. The effect of the system is to make emulation invisible. A user requests a double-precision matrix multiplication, and this is what they get. The major problem here is that the heuristic used to make a decision, given ESC, is left completely opaque. The paper just says that such a heuristic is used. The system is at least designed in such a way that degraded *accuracy* should not be possible, but we don't know if we're leaving performance on the table.

# Evaluation

## Experimental Setting

Inputs: Same synthetic matrices as before
Hardware: GB200, RTX Pro 6000 Blackwell Server Edition
Error measurement: Maximum componentwise relative error, computed against a double-double reference.
Variants: With and without the fallback -- i.e. always emulating with the slices indicated by ESC or only using emulation when it is deemed performant

## Results

Accuracy: Measured using tests proposed in a preprint by Demmel et al. (2026). The argument is that, emulation with ADP passes the tests which suggest that the system is indistinguishable from other $O(n^3)$ matrix multiplication schemes, and that emulation with ADP passes the most stringent accuracy criteria imposed on matrix multiplication algorithms.

Throughput: Speedups of up to ~2x on GB200 (report hardware ratios), up to 11x on RTX Pro 6000 Blackwell SE (report hardware ratios)

Performance breakdown: Most of the time is spent in matrix multiplication, and overhead from calculating ESC, decision-making logic, and splitting are negligible

# Discussion

## Purpose of the Ozaki scheme/What the evaluations tell us

The Ozaki scheme is now being used as an explicit alternative to FP64 matrix multiplication. But the evaluations remain in the limited form that they were back when it was being evaluated as an accurate matrix multiplication scheme. On some level, this is okay, because if inputs that perform badly under emulation come along, ADP should reject emulation altogether. But given that we don't actually know how the heuristic that makes that decision works, and we don't see the system functioning on particularly adversarial cases, there's pretty big gap being left.

Also the performance results just reinforce that speedups are largely hardware dependent.

Also we're still missing a proper look at memory overhead.

## Implementation analysis

The implementation makes changes to the Ozaki scheme such that the original exactness property of slice products is no longer guaranteed. In the last scheme, while the truncation of remainders meant that accuracy could actually degrade to be worse than FP64 matrix multiplication, now we don't even have the guarantee that slice products are accurate. To be fair, this should only come into play for very large matrices; we can calculate the exact inner dimension $n$ for which $\alpha$ falls below 8. But it is still a change.

On the other hand, the scheme is now ekeing out the best memory overhead it could possibly achieve, so there's that.

Obviously this is much more usable than previous impolementations, but with a huge asterisk that, again, the decision making of the ADP system is a black box.

