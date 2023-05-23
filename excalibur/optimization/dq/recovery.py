import logging
import time
from typing import List

import numpy as np

from ..qcqp import DualRecoveryResult, is_global, QCQPDualResult, QCQPRelaxedResult, QuadraticFun, solve_qcqp, \
    SDRRecoveryResult
from excalibur.utils.math import sorted_eig


def recover_from_dual(dual_result: QCQPDualResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                      real_indices: np.ndarray, dual_indices: np.ndarray, optimize: bool = False,
                      eps_constraints: float = 1e-5, eps_gap: float = 1e-5) -> DualRecoveryResult:
    # prepare result
    result = DualRecoveryResult()

    # nullspace dimensions
    if optimize:
        nullspace_dims = [1, 2, dual_result.Z.shape[0]]
    else:
        nullspace_dims = [1, 2]

    # start time
    start_time = time.time()

    # sorted eigen decomposition for nullspace
    vals, vecs = sorted_eig(dual_result.Z)

    # iterate nullspace dimensions
    for ns_dim in nullspace_dims:
        # calculate or optimize factors
        nullspace = vecs[:, :ns_dim]
        if ns_dim == 1:
            factors = calculate_nullspace_factors_1d(nullspace, real_indices)
        elif ns_dim == 2:
            factors = calculate_nullspace_factors_daniilidis(nullspace, real_indices, dual_indices)
        else:
            factors = optimize_nullspace_factors(nullspace, Q, constraint_funs[:ns_dim], real_indices)

        if factors is None:
            continue
        solution_candidate = nullspace @ factors

        # check primal feasibility
        max_constraint_error = np.max([np.abs(fun.eval(solution_candidate)) for fun in constraint_funs])
        if max_constraint_error > eps_constraints:
            continue

        # check duality gap
        primal_value = solution_candidate.T @ Q @ solution_candidate
        duality_gap = primal_value - dual_result.value
        if result.duality_gap is None or np.abs(duality_gap) < result.duality_gap:
            result.success = True
            result.value = primal_value
            result.x = solution_candidate
            result.duality_gap = duality_gap
            result.ns_dim = ns_dim

            # globally optimal solution found with zero duality gap
            if is_global(dual_result.value, duality_gap, eps_gap):
                result.is_global = True
                break

    # stop time
    result.run_time = time.time() - start_time

    # return result
    return result


def recover_from_sdr(relaxed_result: QCQPRelaxedResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                     real_indices: np.ndarray, assume_rank1: bool = False, eps_rank: float = 1e-3,
                     eps_constraints: float = 1e-5, eps_gap: float = 1e-5) -> SDRRecoveryResult:
    # initialize result
    result = SDRRecoveryResult()

    # start time
    start_time = time.time()

    # matrix space
    U, S, _ = np.linalg.svd(relaxed_result.X)
    is_non_zero = np.abs(S) > eps_rank
    result.rank = np.sum(is_non_zero)

    # check rank
    if assume_rank1:
        if result.rank != 1:
            logging.warning(f"Rank is assumed to be one even though it is {result.rank}")
        space = U[:, [np.argmax(S)]]
    elif result.rank == 1:
        space = U[:, is_non_zero]
    else:
        result.message = f"Cannot recover primal solution: rank {result.rank} is greater than one"
        return result

    # calculate primal solution
    factors = calculate_nullspace_factors_1d(space, real_indices)
    solution_candidate = space @ factors

    # check primal feasibility
    max_constraint_error = np.max([np.abs(fun.eval(solution_candidate)) for fun in constraint_funs])
    if max_constraint_error > eps_constraints:
        result.message = "Recovered solution is no primal feasible"
        logging.error(result.message)
        return result

    # check gap
    result.value = solution_candidate.T @ Q @ solution_candidate
    result.sdr_gap = result.value - relaxed_result.value
    sdr_gap_rel = result.sdr_gap / relaxed_result.value
    if np.abs(result.sdr_gap) > eps_gap and np.abs(sdr_gap_rel) > eps_gap:
        result.message = f"Relaxation gap is large: {result.sdr_gap}"
        logging.error(result.message)
        return result

    # stop time
    result.run_time = time.time() - start_time

    # store solution
    result.success = True
    result.x = solution_candidate
    result.is_global = True

    return result


def calculate_nullspace_factors_1d(nullspace, norm_one_indices):
    assert nullspace.shape[1] == 1
    nullspace_norm = np.linalg.norm(nullspace[norm_one_indices, :])
    return np.array([1.0 / nullspace_norm])


def calculate_nullspace_factors_daniilidis(nullspace, real_indices, dual_indices):
    assert nullspace.shape[1] == 2
    # assert len(real_indices) == len(dual_indices) == 4

    # split nullspace
    nullspace_real = nullspace[real_indices, :]
    nullspace_dual = nullspace[dual_indices, :]

    # separate real and dual part
    u1 = nullspace_real[:, 0]
    v1 = nullspace_dual[:, 0]
    u2 = nullspace_real[:, 1]
    v2 = nullspace_dual[:, 1]
    order_changed = False

    # change order if necessary
    if abs(u1.T @ v1) < abs(u2.T @ v2):
        u1, u2 = u2, u1
        v1, v2 = v2, v1
        order_changed = True

    # a, b, c of eq 34
    a1 = u1.T @ u1
    b1 = 2 * u1.T @ u2
    c1 = u2.T @ u2

    # a, b, c of eq 35
    a2 = u1.T @ v1
    b2 = u1.T @ v2 + u2.T @ v1
    c2 = u2.T @ v2

    # solve eq 35
    disc2_square = b2 ** 2 - 4 * a2 * c2
    if disc2_square < -1e-6:
        return None
    disc2 = np.sqrt(np.abs(disc2_square))
    s1 = (-b2 + disc2) / (2 * a2)
    s2 = (-b2 - disc2) / (2 * a2)

    # select s corresponding two largest value of eq 34
    value1 = s1 ** 2 * a1 + s1 * b1 + c1
    value2 = s2 ** 2 * a1 + s2 * b1 + c1

    if value1 > value2:
        s = s1
        value = value1
    else:
        s = s2
        value = value2

    # check value
    if value < 1e-6:
        return None

    # calculate lambdas from s
    l2 = np.sqrt(1 / value)
    l1 = s * l2

    # lambdas
    if order_changed:
        return np.array([l2, l1])
    else:
        return np.array([l1, l2])


def optimize_nullspace_factors(nullspace, Q, constraint_funs, one_indices, solver_kwargs=None):
    # adjust optimization problem
    Q_ns = nullspace.T @ Q @ nullspace
    constraint_funs_ns = [QuadraticFun(nullspace.T @ fun.A @ nullspace, fun.c)
                          for fun in constraint_funs]

    # initial value
    x0 = np.zeros(nullspace.shape[1])
    x0[0] = 1.0 / np.linalg.norm(nullspace[one_indices, 0])

    # solve
    opt_result = solve_qcqp(Q_ns, constraint_funs_ns, x0, solver_kwargs)

    # result
    if not opt_result.success:
        return None
    return opt_result.x
