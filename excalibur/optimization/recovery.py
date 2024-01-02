import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from excalibur.utils.logging import logger
from excalibur.utils.math import sorted_eig

from .qcqp import DualRecoveryResult, is_global, QCQPDualResult, QCQPRelaxedResult, QuadraticFun, SDRRecoveryResult
from .utils import MultiThreshold, DEFAULT_CONSTRAINT_EPS, DEFAULT_GAP_EPS, DEFAULT_RANK_EPS


def calculate_nullspace_factors_1d(nullspace: np.ndarray, norm_one_indices: Union[List, np.ndarray],
                                   first_positive: bool = False) -> Optional[np.ndarray]:
    assert nullspace.shape[1] == 1

    # calculate norm
    nullspace_norm = np.linalg.norm(nullspace[norm_one_indices, :])

    # check norm
    if nullspace_norm < 1e-6:
        logger.info("Nullspace factor calculation (1d) failed")
        return None

    # adjust sign
    if first_positive and nullspace[norm_one_indices[0], :] < 0.0:
        nullspace_norm *= -1.0

    # factor
    return np.array([1.0 / nullspace_norm])


def optimize_nullspace_factors(nullspace, Q, constraint_funs, x0, solver_kwargs=None):
    logger.warning("Skipping nullspace factor optimization, as it currently causes segfaults in "
                   "scipy.optimize.minimize")
    return None

    # # adjust optimization problem
    # Q_ns = nullspace.T @ Q @ nullspace
    # constraint_funs_ns = [QuadraticFun(nullspace.T @ fun.A @ nullspace, fun.c)
    #                       for fun in constraint_funs]
    #
    # # solve
    # opt_result = solve_qcqp(Q_ns, constraint_funs_ns, x0, solver_kwargs)
    #
    # # result
    # if not opt_result.success:
    #     return None
    # return opt_result.x


def recover_from_dual(dual_result: QCQPDualResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                      nullspace_factor_fun_dict: Dict[int, Callable[[np.ndarray], np.ndarray]], optimize: bool = False,
                      eps_constraints: Union[float, MultiThreshold] = DEFAULT_CONSTRAINT_EPS,
                      eps_gap: float = DEFAULT_GAP_EPS) -> DualRecoveryResult:
    # prepare threshold
    if not isinstance(eps_constraints, MultiThreshold):
        eps_constraints = MultiThreshold(default=eps_constraints)

    # prepare result
    result = DualRecoveryResult()

    # nullspace dimensions
    nullspace_dims = list(sorted(nullspace_factor_fun_dict.keys()))
    if optimize:
        nullspace_dims.append(dual_result.Z.shape[0])

    # start time
    start_time = time.time()

    # sorted eigen decomposition for nullspace
    vals, vecs = sorted_eig(dual_result.Z)
    result.aux_data['vals'] = vals

    # init storage for factors
    best_factors = None
    min_constraint_error_sum = None

    # iterate nullspace dimensions
    result.aux_data['constraint_errors'] = {}
    for ns_dim in nullspace_dims:
        # calculate or optimize factors
        nullspace = vecs[:, :ns_dim]
        if ns_dim in nullspace_factor_fun_dict:
            factors = nullspace_factor_fun_dict[ns_dim](nullspace)
        else:
            # initial value
            if best_factors is None:
                continue
            x0 = np.zeros(nullspace.shape[1])
            x0[:len(best_factors)] = best_factors

            # optimize
            factors = optimize_nullspace_factors(nullspace, Q, constraint_funs, x0)

        if factors is None or np.any(np.isnan(factors) | np.isinf(factors)):
            continue
        solution_candidate = nullspace @ factors

        # calculate constraint errors
        constraint_errors = [np.abs(fun.eval(solution_candidate)) for fun in constraint_funs]
        result.aux_data['constraint_errors'][ns_dim] = constraint_errors

        # store factors and solution candidate for lowest constraint errors (for later optimization)
        constraint_error_sum = np.sum(constraint_errors)
        if best_factors is None or constraint_error_sum < min_constraint_error_sum:
            best_factors = factors
            min_constraint_error_sum = constraint_error_sum
            if not result.success:
                result.x = solution_candidate

        # check primal feasibility
        if np.any([err > eps_constraints.get(idx) for idx, err in enumerate(constraint_errors)]):
            continue

        # check duality gap
        primal_value = solution_candidate.T @ Q @ solution_candidate
        duality_gap = primal_value - dual_result.value
        if result.duality_gap is None or np.abs(duality_gap) < np.abs(result.duality_gap):
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

    # set message
    if not result.success:
        if len(result.aux_data['constraint_errors']) == 0:
            result.message = "All nullspace recovery strategies failed"
        else:
            result.message = "No primal feasible solution found"

    return result


def recover_from_sdr(relaxed_result: QCQPRelaxedResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                     norm_one_indices: Union[List, np.ndarray], assume_rank1: bool = False,
                     eps_rank: float = DEFAULT_RANK_EPS, eps_constraints: float = DEFAULT_CONSTRAINT_EPS,
                     eps_gap: float = DEFAULT_GAP_EPS)\
        -> SDRRecoveryResult:
    # prepare threshold
    if not isinstance(eps_constraints, MultiThreshold):
        eps_constraints = MultiThreshold(default=eps_constraints)

    # initialize result
    result = SDRRecoveryResult()

    # start time
    start_time = time.time()

    # matrix space
    U, S, _ = np.linalg.svd(relaxed_result.X)
    result.aux_data['S'] = S
    is_non_zero = np.abs(S) > eps_rank
    result.rank = np.sum(is_non_zero)

    # check rank
    if assume_rank1:
        if result.rank != 1:
            logger.info(f"Rank is assumed to be one even though it is {result.rank}")
        space = U[:, [np.argmax(S)]]
    elif result.rank == 1:
        space = U[:, is_non_zero]
    else:
        result.message = f"Cannot recover primal solution: rank {result.rank} is greater than one"
        return result

    # calculate primal solution
    factors = calculate_nullspace_factors_1d(space, norm_one_indices)
    solution_candidate = space @ factors

    # check primal feasibility
    constraint_errors = [np.abs(fun.eval(solution_candidate)) for fun in constraint_funs]
    if np.any([err > eps_constraints.get(idx) for idx, err in enumerate(constraint_errors)]):
        if result.message:
            logger.warning("Recovered solution is not primal feasible")
        else:
            result.message = "Recovered solution is not primal feasible"
            return result

    # store solution
    result.success = True
    result.x = solution_candidate

    # check relaxation gap for globality
    result.value = solution_candidate.T @ Q @ solution_candidate
    result.sdr_gap = result.value - relaxed_result.value
    sdr_gap_rel = result.sdr_gap / relaxed_result.value
    result.is_global = np.abs(result.sdr_gap) < eps_gap or np.abs(sdr_gap_rel) < eps_gap

    # stop time
    result.run_time = time.time() - start_time

    return result
