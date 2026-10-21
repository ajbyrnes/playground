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