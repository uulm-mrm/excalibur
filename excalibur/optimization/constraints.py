import numpy as np

from .qcqp import QuadraticFun


def homogeneous_constraint(dim, hom_index):
    A = np.zeros([dim, dim])
    A[hom_index, hom_index] = -1.0
    return QuadraticFun(A, 1.0)


def parallel_constraints_4d(dim, indices1, indices2, reduced=False):
    assert len(indices1) == len(indices2) == 4

    if reduced:
        index_pairs = [(0, 1), (0, 2), (0, 3)]
    else:
        index_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    P_list = []

    for j, k in index_pairs:
        P = np.zeros([dim, dim])
        P[indices1[j], indices2[k]] = 1
        P[indices2[k], indices1[j]] = 1
        P[indices1[k], indices2[j]] = -1
        P[indices2[j], indices1[k]] = -1
        P_list.append(QuadraticFun(P))

    return P_list


def norm_constraint(dim, indices, translation_norm, norm_one_indices=None):
    P = np.zeros((dim, dim))
    P[indices, indices] = -1.0
    c = translation_norm ** 2

    if norm_one_indices is None:
        return QuadraticFun(P, c=c)
    else:
        P[norm_one_indices, norm_one_indices] = c
        return QuadraticFun(P)
