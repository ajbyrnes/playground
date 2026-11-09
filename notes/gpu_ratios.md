# GPUs covered in (or relevant to) the WCP / talk

Fill in peak dense throughput (no sparsity) to get the low-precision : FP64 ratio.
For NVIDIA, FP64 has two numbers on many parts: vector (CUDA cores) and Tensor Core. Note which one the ratio uses.
Ootomo et al.'s ratios on the slides use FP64 Tensor Core where it exists (A100) and FP64 vector otherwise.

Ratios are all TFlops/TOPs:  
  
| Hardware | TFLOPs : TOPs | Emulation speedup? | 
| ----- | ----- | ------ | 
| A100 | 1:32 | No | 
| RTX Titan | 1:522 | Yes | 
| RTX A6000 | 1:515 | Yes | 
| RTX 6000 Ada | 1:520 | Yes | 
| H100 | 1:32 | No |
| GB10 | 1:474 | Yes | 
| GB200 | 1:128 | Yes | 
| RTX Pro 6000 Blackwell SE | 1:519 | Yes | 
| MI250X | 1:4 | Predict No |
| MI300A | 1:16 | Predict No | 
| MI300X | 1:16 | Predict No | 
| MI355X | 1:64 | Unclear |