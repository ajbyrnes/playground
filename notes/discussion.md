# Discussion and Conclusion

## Tracing the Ozaki scheme

Each work moved the scheme one step. Ozaki et al. established it as a viable method for accurate matrix multiplication, outperforming alternatives like XBLAS at comparable accuracy. Ootomo et al. showed that integer Tensor Cores can push it past native FP64 throughput on consumer GPUs, though not on datacenter chips with strong FP64. Schwarz et al. showed it can be deployed safely through ADP. Brower et al. confirmed the results carry over to at least one real scientific application, DMRG, with sustained accuracy across hundreds of iterations.



## Back to the debate

Can emulation boost FP64 throughput? Yes, sometimes. To be fair, even the most pro-emulation position is not suggesting that emulation can just replace all FP64 performance. Broadly speaking, the highly pro-emulation argument is that workloads that are compute-bound on matrix multiplication would benefit from emulation, providing the low-precision throughput is there to support them, and that workloads that are memory-bound wouldn't benefit from emulation, but they also wouldn't necessarily benefit from higher FP64 throughput. We do see this play out, to an extent, in the results of Brower et al. Emulation does not accelerate their workload even on a chip where the hardware favors it. Brower et al. describe their workload's matrix multiplication steps as being memory bound -- but also, Schwarz et al.'s results suggest that emulation would still not provide speedup on their problem size, because at low problem sizes, the overhead of the ADP mechanisms and the matrix product transformation is more significant.

## Performance comes from hardware / When is emulation useful?

(This is going to be a little unorgranized, unfortunately. We've reached the edge of the literature -- this is the main limitation of my problem selection lol.)

The number one factor determining whether the Ozaki scheme is performant is the FP64-to-low-precision throughput ratio of the target hardware. Emulation becomes faster than native DGEMM precisely on the hardware where native FP64 has been deprioritized.

The cynical take on this is that, well, you can make emulation arbitrarily superior to FP64 by just having terrible FP64 support. But workloads that genuinely need high FP64 throughput would be better off running on chips that actually have high FP64 throughput. The more intriguing perspective is that low-precision hardware is cheaper to produce and run than high-precision hardware. If the same chip-area can support higher *emulated* throughput than native throughput, then maybe we're just being very inefficient by clinging to designs that load up on native FP64 hardware.

(At the very least, this means that we can squeeze some FP64 performance out of low-precision hardware, extending the capabilities of that hardware if it is all that is available.)

The problem with that view is that, while hardware is the primary factor that determines whether emulation can speedup double-precision matrix multiplication, it is not the *only* factor. Speedup is also dependent on the *inputs*.  We have seen that, when hardware favors emulation, emulation can be faster *on some inputs*.  We know for a fact that emulation can be enormously beneficial on some hardware, and on some specific inputs. 

## Future work

Framing this as questions that we can ask about emualtion:

For a given combination of inputs and hardware, should we use emulation? This can be tested empirically, or we can make predictions using estimators like ESC. But the estimators are calculated based on estimates of the highest magnitude value in the result, and the highest magnitude values in the inputs. They do not account for matrix structure which may lead to heavier cancellation -- or matrix size/shape, whose impacts must be understood more heuristically.

For some particular inputs, what hardware will carry out the workload most efficiently? This gets at an important aspect of emulation that was brought up a couple minutes ago: would it be more efficient to just load chips with low-precision hardware? The thing is that low-precision Tensor Cores are built for matrix multiplication specifically; workloads that involve lots of FP64 computation that is *not* matrix multiplication may still prefer hardware with lots of actual native FP64 support. 

Both of the above questions are answered with broad characterization studies on matrices with different ESC values, structures, and shapes. For the most part, we've just seen Ozaki implementations being evaluated with the same synthetic matrices that stress magnitude range and size only.

What are the impacts of memory overhead? Not just the straightforward "how much more memory needs to be allocated" problem, but "how is data movement in my application affected by blowing up the working set with matrix slices"?  
   

## A variable-precision method

This is the main argument of the talk. The scheme was framed first as a way to get more accurate products, then as a replacement for FP64. It's better understood as a variable-precision method. In the 2012 formulation, the slices live inside FP64's fixed 53-bit mantissa, so the slice count has a ceiling. With integer slices, that ceiling is gone. Each additional 8-bit slice extends the represented mantissa by another 8 bits, and nothing requires that number to match 53. Schwarz et al. hint at this by describing ESC in terms of mantissa bit coverage, not FP64-equivalence. Brower et al. show it in practice: 4 and 6 slices are less than FP64 and still give chemical accuracy in DMRG. The same knob that dials precision below FP64 dials it above FP64 too.  
  
Instead of asking whether emulation can replace FP64 for a workload, ask what precision that workload actually needs. The mixed-precision computing community has been developing tools to answer that question for over a decade. Placing the Ozaki scheme in that body of work may help identify its most effective role in scientific computing.
  
TL;DR: Major point: The Ozaki scheme gives us a framework to study variable precision methods, and how much precision a particular problem really needs.  

## Future work

Each gap comes from something earlier in the talk:

- **How the scheme fails.** We have very little understanding of emulation on structured matrices, real-application matrices, or extreme aspect ratios. There may be some domain that frequently works with matrices that are adversarial to emulation.
- **Memory overhead.** Acknowledged as a concern throughout the literature, rarely reported in detail, and not yet measured in a full application.
- **Slice count below the FP64 bound.** ADP makes a binary decision with the accuracy target fixed to FP64. A runtime that picks the slice count from application-level accuracy requirements would widen the scheme's useful range considerably. This is the variable-precision framing turned into a tool.
- DO NOT INCLUDE THIS ANYMORE: **Lightweight heuristics.** ESC, even block-wise, adds measurable overhead to every multiply. The same decision could perhaps be made from cheaper signals: matrix dimensions, hardware identifiers, input patterns. This goes hand in hand with understanding failure patterns.
- **Revisiting older techniques.** The Ozaki scheme was impractical on throughput for over a decade until the hardware arrived. Other techniques in the numerical linear algebra literature may map well onto the asymmetric throughput of modern GPUs.


## Conclusion

We traced the scheme from its origins as a method for accurate matrix multiplication, through integer Tensor Cores, its integration into cuBLAS with runtime safety guarantees, and its first use inside a full scientific application. For each work, we looked at the breadth of its evaluation and at the performance and usability of its implementation. Close with Joseph George's quote from the intro: "right answers fast." What counts as a right answer depends on the precision the workload needs, and the slice count is how the Ozaki scheme lets you choose it.
