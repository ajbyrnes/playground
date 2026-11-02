import math

import numpy as np

def extract_scalar(x, M):
    sigma = 2**M * 2**(np.ceil(np.log2(np.abs(x))))
    assert(np.abs(x) <= sigma)
    x_hi = sigma + x - sigma
    x_lo = x - x_hi
    return x_hi, x_lo

def extract_vector(x, M):
    max_x = np.max(np.abs(x))
    sigma = 2**M * 2**(np.ceil(np.log2(max_x)))
    assert(max_x <= sigma)
    x_hi = sigma + x - sigma
    x_lo = x - x_hi
    return x_hi, x_lo

def split_vector(x, M):
    """Split x into slices with x = slices[0] + slices[1] + ... exactly.

    Split_Mat (Algorithm 3) with no limit on the number of slices: each pass
    of extract_vector keeps the top bits of x as a slice and hands the rest
    to the next pass, until nothing is left.
    """
    x = np.asarray(x, dtype=np.float64)
    slices = []
    while np.any(x != 0):
        x_hi, x = extract_vector(x, M)
        slices.append(x_hi)
    return slices

def beta(n):
    """Equation (6): β = ⌈(−log2 u + log2 n) / 2⌉ with u = 2^-53.

    β takes the place of M. Each slice then holds α = 53 − β bits, few enough
    that a dot product of length n between two slices can't round.
    """
    return math.ceil((53 + math.log2(n)) / 2)

def split_mat(A, max_slices=math.inf):
    """Algorithm 3 (Split_Mat), without the sparse storage: A = D[0] + D[1] + ... exactly.

    Each row gets its own σ from its own max |a_ij|, so every row of D[k] is
    what split_vector(row, β) would give as its k-th slice. With max_slices,
    whatever is left after max_slices - 1 passes goes in the last slice.
    """
    A = np.array(A, dtype=np.float64)  # a copy: A is overwritten with the remainder
    b = beta(A.shape[1])
    D = []
    while len(D) < max_slices - 1:
        mu = np.max(np.abs(A), axis=1, keepdims=True)  # μ_i = max_j |a_ij|, as a column
        if np.all(mu == 0):
            return D
        # rows that are already 0 get σ = 2^-inf = 0, which leaves them 0
        with np.errstate(divide="ignore"):
            sigma = 2.0**b * 2.0**np.ceil(np.log2(mu))
        D.append((A + sigma) - sigma)  # σ broadcasts across each row: S = σ·eᵀ
        A = A - D[-1]
    D.append(A)
    return D

def eft_mul(A, B):
    """Algorithm 4 (EFT_Mul): AB = C[0] + C[1] + ... exactly, as an unevaluated sum.

    A is split by rows and B by columns (split B's transpose by rows), then
    every slice of A is multiplied by every slice of B. Each fl(D_r @ E_s)
    is exact, so no product rounds; only adding up the C's would.
    """
    D = split_mat(A)
    E = [E_s.T for E_s in split_mat(np.asarray(B).T)]
    return [D_r @ E_s for D_r in D for E_s in E]