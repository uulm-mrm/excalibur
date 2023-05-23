import time
from typing import List

import numpy as np

from ..qcqp import DualRecoveryResult, QuadraticFun, QCQPDualResult
from excalibur.utils.math import sorted_eig


def normalize_rotation_matrix(R):
    # determine epsilon
    eps = np.finfo(R.dtype).eps

    # unit determinant
    det = np.linalg.det(R)
    if np.abs(det) < eps:
        return None
    R *= np.cbrt(np.copysign(1.0, det) / np.abs(det))

    # make orthogonal
    u, _, vt = np.linalg.svd(R)
    R = u @ vt

    # fix inversion
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt

    return R


def recover_from_dual(dual_result: QCQPDualResult, Q: np.ndarray, constraint_funs: List[QuadraticFun], hom_index: int,
                      eps_constraints: float = 1e-4, eps_gap: float = 1e-5) -> DualRecoveryResult:
    # prepare result
    result = DualRecoveryResult()

    # start time
    start_time = time.time()

    # sorted eigen decomposition for nullspace
    vals, vecs = sorted_eig(dual_result.Z)

    # assume one-dimensional nullspace
    nullspace = vecs[:, 0]

    # solution candidate
    solution_candidate = nullspace / nullspace[hom_index]

    # check primal feasibility
    max_constraint_error = np.max([np.abs(fun.eval(solution_candidate)) for fun in constraint_funs])
    if max_constraint_error > eps_constraints:
        return result

    # create result
    result.run_time = time.time() - start_time
    result.success = True
    result.value = solution_candidate.T @ Q @ solution_candidate
    result.x = solution_candidate
    result.duality_gap = result.value - dual_result.value
    result.ns_dim = 1

    # check duality gap
    duality_gap_rel = result.duality_gap / dual_result.value
    result.is_global = np.abs(result.duality_gap) < eps_gap or np.abs(duality_gap_rel) < eps_gap

    return result
