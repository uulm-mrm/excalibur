from typing import List

import numpy as np

from ..qcqp import QuadraticFun
from excalibur.utils.math import canonical_vector, canonical_matrix, cross_product_matrix


def _orthonormal_mat(i, j):
    x = np.zeros((3, 3))
    x[i, j] = 1.0
    x[j, i] = 1.0
    return x


def _orthonormal_constraint_mat(rot_con, hom_con):
    x = np.zeros((10, 10))
    x[:9, :9] = rot_con
    x[9, 9] = hom_con
    return x


def _handedness_constraint_mat(l, i, j, k):
    e_ij = canonical_matrix(3, i, j)
    e_k = canonical_vector(3, k)
    l_d_ijk = canonical_vector(3, l)

    x = np.zeros((10, 10))
    x[:9, :9] = - np.kron(e_ij, cross_product_matrix(l_d_ijk))
    x[:9, 9] = - np.kron(e_k, l_d_ijk)
    return x


def rotmat_constraints_hom() -> List[QuadraticFun]:
    # orthonormal constraints
    ortho_matrices = [_orthonormal_mat(i, j) for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]]
    A_rows = [_orthonormal_constraint_mat(np.kron(-np.eye(3), m), np.trace(m)) for m in ortho_matrices]
    A_cols = [_orthonormal_constraint_mat(np.kron(-m, np.eye(3)), np.trace(m)) for m in ortho_matrices]

    # handedness constraints
    A_hand = [_handedness_constraint_mat(l, i, j, k) for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] for l in range(3)]
    A_hand_sym = [x + x.T for x in A_hand]

    # homogenization constraint
    A_hom = np.zeros((10, 10))
    A_hom[9, 9] = -1.0

    # constraint functions
    constraint_funs = [QuadraticFun(A) for A in [*A_rows, *A_cols, *A_hand_sym]]
    constraint_funs.append(QuadraticFun(A_hom, 1.0))
    return constraint_funs
