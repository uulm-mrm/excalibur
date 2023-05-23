from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Optional

import numpy as np

from .recovery import calculate_nullspace_factors_1d, recover_from_dual, recover_from_sdr
from ..qcqp import DualRecoveryResult, is_global, QCQPDualResult, QCQPRelaxedResult, QCQPResult, QuadraticFun, \
    SDRRecoveryResult, solve_qcqp, solve_qcqp_dual, solve_qcqp_sdr, verify_qcqp_result_globality
from excalibur.utils.math import sorted_eig
from excalibur.utils.parameters import add_default_kwargs


class QCQPDQMethod(Enum):
    NONE = auto()
    QCQP = auto()
    DUAL = auto()
    SDR = auto()


@dataclass
class QCQPDQResult:
    success: bool = False
    x: Optional[np.ndarray] = None
    gap: Optional[float] = None
    is_global: bool = False
    method: QCQPDQMethod = QCQPDQMethod.NONE
    sdr_result: Optional[QCQPRelaxedResult] = None
    sdr_recovery: Optional[SDRRecoveryResult] = None
    dual_result: Optional[QCQPDualResult] = None
    dual_recovery: Optional[DualRecoveryResult] = None
    primal_result: Optional[QCQPResult] = None


def solve_qcqp_dq(Q: np.ndarray, constraint_funs: List[QuadraticFun], real_indices: np.ndarray,
                  dual_indices: np.ndarray, x0: Optional[np.ndarray] = None,
                  use_sdr: bool = True, use_dual: bool = True, use_qcqp: bool = True, qcqp_first: bool = False,
                  sdr_kwargs: Optional[Dict] = None, sdr_rec_kwargs: Optional[Dict] = None,
                  dual_kwargs: Optional[Dict] = None, dual_rec_kwargs: Optional[Dict] = None,
                  qcqp_kwargs: Optional[Dict] = None, qcqp_ver_kwargs: Optional[Dict] = None):
    # default arguments
    sdr_kwargs = add_default_kwargs(sdr_kwargs)
    sdr_rec_kwargs = add_default_kwargs(sdr_rec_kwargs, assume_rank1=len(constraint_funs) <= 2)
    dual_kwargs = add_default_kwargs(dual_kwargs)
    dual_rec_kwargs = add_default_kwargs(dual_rec_kwargs, optimize=False)
    qcqp_kwargs = add_default_kwargs(qcqp_kwargs)
    qcqp_ver_kwargs = add_default_kwargs(qcqp_ver_kwargs)

    # initialize result
    res = QCQPDQResult()

    # primal problem first
    if use_qcqp and qcqp_first:
        # initial variable
        if x0 is None:
            x0 = np.zeros(Q.shape[0])

        # solve primal problem
        res.primal_result = solve_qcqp(Q, constraint_funs, x0, **qcqp_kwargs)

        # result result
        if res.primal_result.success:
            res.success = True
            res.x = res.primal_result.x
            res.method = QCQPDQMethod.QCQP
            res.is_global = verify_qcqp_result_globality(Q, constraint_funs, res.primal_result, **qcqp_ver_kwargs)

            # stop early if global
            if res.is_global:
                return res

    # semidefinite relaxation
    if use_sdr and len(constraint_funs) <= 2:
        res.sdr_result = solve_qcqp_sdr(Q, constraint_funs, **sdr_kwargs)

        if res.sdr_result.success:
            # sdr recovery
            res.sdr_recovery = recover_from_sdr(res.sdr_result, Q, constraint_funs, real_indices, **sdr_rec_kwargs)

            if res.sdr_recovery.success:
                res.success = True
                res.x = res.sdr_recovery.x
                res.gap = res.sdr_recovery.sdr_gap
                res.is_global = res.sdr_recovery.is_global
                res.method = QCQPDQMethod.SDR

                # stop early if global
                if res.is_global:
                    return res

    # dual problem
    if use_dual:
        res.dual_result = solve_qcqp_dual(Q, constraint_funs, **dual_kwargs)

        if res.dual_result.success:
            # dual recovery
            res.dual_recovery = recover_from_dual(res.dual_result, Q, constraint_funs, real_indices, dual_indices,
                                                  **dual_rec_kwargs)

            if res.dual_recovery.success:
                res.success = True
                res.x = res.dual_recovery.x
                res.gap = res.dual_recovery.duality_gap
                res.is_global = res.dual_recovery.is_global
                res.method = QCQPDQMethod.DUAL

                # stop early if global
                if res.is_global:
                    return res

    # primal problem last
    if use_qcqp and not qcqp_first:
        # initial variable
        if x0 is None:
            if res.dual_result is not None:
                if res.dual_recovery.success:
                    # use non-global recovered solution as initial variable
                    x0 = res.dual_recovery.x
                else:
                    # use recovery strategy for one-dimensional nullspace as initial variable,
                    # even though it might be unfeasible
                    _, ns_vecs = sorted_eig(res.dual_result.Z)
                    nullspace = ns_vecs[:, :1]
                    factors = calculate_nullspace_factors_1d(nullspace, real_indices)
                    x0 = nullspace @ factors

        # emergency initial solution
        if x0 is None:
            x0 = np.zeros(Q.shape[0])

        # solve primal problem
        res.primal_result = solve_qcqp(Q, constraint_funs, x0, **qcqp_kwargs)

        # result result
        if res.primal_result.success:
            # calculate duality gap, if dual result was calculated
            gap = res.primal_result.value - res.dual_result.value if res.dual_result is not None else None

            # compare to recovered dual result and overwrite if primal result is better
            if res.dual_recovery is None or not res.dual_recovery.success or \
                    (gap is not None and np.abs(gap) < np.abs(res.dual_recovery.duality_gap)):
                res.success = True
                res.x = res.primal_result.x
                res.gap = gap
                res.method = QCQPDQMethod.QCQP

                # globality of primal solution
                if gap is None:
                    res.is_global = verify_qcqp_result_globality(Q, constraint_funs, res.primal_result,
                                                                 **qcqp_ver_kwargs)
                else:
                    res.is_global = is_global(res.primal_result.value, gap)

    return res
