import cvxpy as cvx
import numpy as np


def stack_numpy_to_cvx(matrices: np.array, exprs: cvx.Variable) -> cvx.kron:
    """stack numpy matrices while scaling with cvx expression:
    each depth layer of matrices is scaled with the respective
    expression entry in exprs"""

    # each np matrix is scaled by cvx expr, thus, lengths must fit
    assert matrices.ndim == 3 and matrices.shape[2] == exprs.shape[0]
    n = matrices.shape[2]

    # use kron to scale np matrix with scalar expression
    # Important: do not initialize varout with cp.Variable(),
    # it will result in constraints not leading to the same solutions later...
    varout = cvx.kron(matrices[:, :, 0], cvx.reshape(exprs[0], (1, 1)))
    for i in range(1, n):
        varout += cvx.kron(matrices[:, :, i], cvx.reshape(exprs[i], (1, 1)))

    return varout
