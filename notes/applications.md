# Ozaki in Applications

## Early examples

Every evaluation so far has been on synthetic matrices. The obvious question is whether there are matrix products in real applications that benefit from emulation. Ootomo et al. ran quantum circuit simulation (random unitary circuits, which are basically toy models) and got 2–4.85× over cuBLAS ZGEMM with the same or better accuracy. Schwarz et al. put emulated DGEMM into cuSOLVER's QR factorization. QR is an application of matrix multiplication, but it isn't a scientific application. Neither of these tells us what happens inside an actual science code.

## Brower et al. (2026)

This is the first time the Ozaki scheme shows up inside a real scientific code: DMRG for quantum chemistry, run on a DGX B200 with NVIDIA's prerelease cuBLAS emulation. Half the author list is NVIDIA (including Brower and Gunnels, who are also on the Schwarz paper), and the DMRG code is Menczer and Legeza's.

## DMRG

DMRG is an iterative method for finding the lowest-energy state (ground state) of a molecule. The wavefunction is stored as a chain of tensors, one per orbital, each of size $D \times 4 \times D$. The bond dimension $D$ is the accuracy knob: larger $D$ is more accurate, and the exact answer comes back if $D$ is allowed to grow exponentially. Memory scales like $N^2 D^2$ and compute like $N^4 D^3$, where $N$ is the number of orbitals.

Each iteration has three main pieces: tensor contractions (matrix multiplications), an iterative eigensolver (Lánczos or Davidson) on an effective Hamiltonian, and an SVD to move to the next site. The eigensolver and the SVD are the parts that are sensitive to rounding. The eigensolver is about 80–85% of the run time.

Because DMRG is variational, the energy always approaches the true answer from above. So if emulation breaks something, you can see it: the energy goes *below* the exact answer, which is physically impossible. That makes DMRG a nice test bed for emulation accuracy.

## From $D$ to DGEMMs

This is the part that answers "what does $D$ actually mean for DGEMM size."

$D$ is the size of the full matrices, but the code never multiplies full $D \times D$ matrices. The molecule has symmetries (particle number, spin), and those split every operator and every piece of the wavefunction into blocks labeled by quantum numbers (sectors). Only certain row-sector/column-sector pairs can be nonzero, and each nonzero block is stored as its own dense matrix. All of the linear algebra is done block by block. So the matrices that actually get handed to DGEMM are sector blocks, which are pieces of $D$, not $D$ itself.

The details are in Menczer & Legeza's hybrid CPU-GPU paper (JCTC 2025, Eq. 2 and Algorithm 2). In the eigensolver, applying the effective Hamiltonian to the wavefunction becomes a long list of independent tasks of the form

$$B \mathrel{+}= \alpha \sum_{i=1}^{p} L_i A R_i^T$$

where $A$ and $B$ are sector blocks of the wavefunction ($A$ is $m \times n$, $B$ is $q \times r$), $L_i$ and $R_i$ are sector blocks of left- and right-block operators ($q \times m$ and $r \times n$), $p$ is the number of operator terms in the task, and $\alpha$ is a precomputed scalar.

Each task is two cuBLAS calls:

1. `cublasDgemmStridedBatched` with $(m, n, k) = (m, r, n)$, batch count $p$: computes $A R_i^T$ for every $i$, with the outputs interleaved in memory.
2. `cublasDgemm` with $(m, n, k) = (q, r, m p)$: multiplies the concatenated $L$ (a $q \times mp$ matrix) by the stacked results, which does the sum over $i$ inside the multiply.

So the DGEMM dimensions are sector sizes $m, n, q, r$, each at most $D$, plus an inner dimension $mp$ that can be bigger than any single sector. There are millions of these tasks per sweep (the papers say "several million independent operations"). The papers say the number of sectors grows with $D$ at first, then once all sectors are populated only the sector sizes grow. They never give the actual sector sizes or a distribution of DGEMM sizes, in any of the four papers I checked (Brower 2026, Menczer et al. 2024 on DGX H100, Menczer & Legeza 2025 hybrid, Menczer & Legeza 2023 SU(2)).

In the SU(2) version of the code the blocks are stored in terms of multiplets, and each operation just picks up an extra precomputed scalar (the GPU kernel barely changes), so the same picture holds with $D$ counted in multiplets.

## (Backup, no slide) How big is $D$?

There is no direct translation from $D$ to a DGEMM size: the blocks are never reported, so the most you can say is that every block dimension is at most $D$. These numbers are here in case someone asks how much work a given $D$ is.

Brower et al. report $D$ in SU(2) multiplets. Each multiplet stands for several actual states, so the equivalent U(1) bond dimension is about 2.3–2.7× larger. The table is from Menczer & Legeza's SU(2) paper (arXiv:2309.16724, Table 1). Its caption doesn't name the system; the surrounding text is about F$_2$ in CAS(18,18), which is the same small system as Brower's Figure 1, so that's almost certainly it. "Largest eigensolver step" is their CC$_\text{max}$: the total work in the most expensive Davidson step at that $D$ (the column header says GFLOP; the caption says FLOP, which looks like a typo).

| $D$ (SU(2)) | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| $D$ (U(1), effective) | 2361 | 4910 | 10928 | 20797 |
| Largest Davidson step | 680 GFLOP | 4,196 GFLOP | 27,586 GFLOP | 193,949 GFLOP |

Brower's Figure 5 gives the same conversion on its labels: for F$_2$, $D = 3$k, 4k, 6k, 8k, 10k SU(2) are 7.0k, 10.6k, 15.7k, 21.7k, 25.5k in U(1).

For a sense of scale: 194 TFLOP is about the work of one square DGEMM of size 46,000. But that work arrives as millions of DGEMMs on sector blocks that are each smaller than $D$, so the individual DGEMMs are nowhere near the sizes where Schwarz et al. or Ootomo et al. saw speedups.

## Accuracy

F$_2$ (small system, exact answer known): six slices reproduce FP64. Four slices are fine. Three slices just barely reach chemical accuracy (1.6 milli-Hartree). Two slices (about half precision) break DMRG: the energy oscillates, iterations go unstable, and some energies go below the exact answer, which shouldn't be possible. Also, the expected improvement with larger $D$ only shows up with four or more slices. N$_2$ looks the same.

Cytochrome P450, CAS(63,58) (large system): four and six slices agree with native FP64. With six slices, the error is around $10^{-5}$ at $D = 1024$ and drops to $10^{-6}$–$10^{-7}$ at larger $D$. Three slices miss milli-Hartree accuracy at every $D \ge 1024$. FeMoco gives similar results.

They also looked at which part of DMRG is sensitive. The Lánczos eigensolver is not sensitive (same iteration count for three or more slices). Davidson is a bit less stable. The SVD/eigendecomposition step is where small slice counts (2–3) really hurt.

Note: all accuracy runs use eager mode with a fixed slice count, not ADP's automatic choice. With the automatic setting on a newer prerelease library, they say energies were slightly more accurate still.

## Performance

Figure 4 is cumulative eigensolver time for the F$_2$ runs. Eager mode emulates every DGEMM regardless of whether it's worth it, and it's much slower than native FP64 for three or more slices. Performant mode lets the library decide per DGEMM (the ADP-style heuristic, an early version of it), and it basically matches FP64, slightly faster with four or six slices at $D = 8192$.

That fits the sector picture. Brower et al. say directly: "While emulation is powerful for compute-bound matrix-multiplications, it is often slower for memory-bound matrix multiplications." Sector DGEMMs are small, so many of them are memory-bound, and eager mode pays the slicing overhead on all of them.

## Performance at scale

Figure 5 is the best TFLOPS of the eigensolver vs. $D$ for five systems: native FP64 on B200 (solid), performant emulation on B200 (dashed), and native FP64 on H100 (dotted). Performant mode tracks native FP64 almost exactly, with small gains only for the largest $D$ and system sizes. Native FP64 on B200 is actually 10–15% *slower* than H100 at large $D$, which is the FP64 trend from the intro showing up in a real code. Emulation, at this point, just about makes up the difference.

For reference, the B200 is 1 : 125 FP64 : INT8 (see notes/gpu_ratios.md), much less lopsided than the RTX Pro 6000.

## What the application tells us

Accuracy: four to six slices are enough for chemical accuracy in real quantum chemistry, and DMRG's variational property makes failures obvious. Two slices are not acceptable.

Speed: no meaningful gain yet. The big DGEMM speedups from the earlier papers depended on large matrices, and here the DGEMMs are sector blocks. Total work per step is huge, but it's split into millions of small multiplications.

Caveats: this is a prerelease cuBLAS, the performance heuristic is an early version, and the paper explicitly defers the real performance study to a later publication. We still don't know what sector sizes look like, so we can't say how much of the workload is in the size range where emulation wins.

## Takeaways

Role: FP64 emulation inside a real scientific code, with the library deciding DGEMM by DGEMM whether to emulate.

Trade-off: accuracy holds with few slices, but speedup depends on DGEMM size, and in DMRG that's set by sector sizes, not $D$.
