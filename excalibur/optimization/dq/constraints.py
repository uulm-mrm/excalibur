import numpy as np

from ..qcqp import QuadraticFun


def dq_real_constraint(dim, real_indices, norm_one_indices=None):
    assert len(real_indices) == 4
    P_real = np.zeros((dim, dim))
    P_real[real_indices, real_indices] = -1.0

    if norm_one_indices is None:
        return QuadraticFun(P_real, c=1.0)
    else:
        P_real[norm_one_indices, norm_one_indices] = 1.0
        return QuadraticFun(P_real)


def dq_dual_constraint(dim, real_indices, dual_indices):
    assert len(real_indices) == len(dual_indices) == 4
    P_dual = np.zeros((dim, dim))
    P_dual[real_indices, dual_indices] = 1.0
    P_dual[dual_indices, real_indices] = 1.0
    return QuadraticFun(P_dual)


def dq_constraints(dim, real_indices, dual_indices, norm_one_indices=None):
    assert len(real_indices)
    return [dq_real_constraint(dim, real_indices, norm_one_indices),
            dq_dual_constraint(dim, real_indices, dual_indices)]


def dq_translation_norm_constraint(dim, dual_indices, translation_norm, norm_one_indices=None):
    assert len(dual_indices) == 4

    P = np.zeros((dim, dim))
    P[dual_indices, dual_indices] = -1.0
    c = (translation_norm ** 2) / 4.0

    if norm_one_indices is None:
        return QuadraticFun(P, c=c)
    else:
        P[norm_one_indices, norm_one_indices] = c
        return QuadraticFun(P)
