import numpy as np


DQ_PLANAR_INDICES = np.array([0, 3, 5, 6])


def dq_reduce_to_planar(x: np.ndarray):
    assert (x.ndim == 1 and len(x) == 8) or (x.ndim == 2 and x.shape == (8, 8))

    if x.ndim == 1:
        return x[DQ_PLANAR_INDICES]
    else:
        return x[DQ_PLANAR_INDICES, :][:, DQ_PLANAR_INDICES]


def dq_recover_from_planar(x: np.ndarray):
    assert x.ndim == 1 and len(x) == 4

    dq = np.zeros(8)
    dq[DQ_PLANAR_INDICES] = x
    return dq
