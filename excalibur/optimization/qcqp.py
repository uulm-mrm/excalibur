from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
import scipy.optimize
import time

from excalibur.utils.parameters import add_default_kwargs
from excalibur.utils.logging import logger

from .cvx_utils import stack_numpy_to_cvx
from .linear import solve_linear_problem


@dataclass
class QuadraticFun:
    A: np.ndarray
    c: float = 0.0

    def eval(self, x: np.ndarray):
        return x.T @ self.A @ x + self.c

    def jac(self, x: np.ndarray):
        return 2 * x.T @ self.A


def generate_quadratic_cost_matrix(Mlist: List[np.ndarray], weights: Optional[Union[np.ndarray, List]] = None,
                                   normalize: bool = False) -> Optional[np.ndarray]:
    # check Mlist
    if len(Mlist) == 0:
        return None

    # prepare weights
    if weights is not None:
        if isinstance(weights, np.ndarray):
            if weights.ndim == 1 and len(weights) == len(Mlist):
                # single weight scalar for each sample
                pass
            elif weights.ndim == 1 and len(weights) == Mlist[0].shape[0]:
                # diagonal vector of weight matrix for each sample
                weights = [np.diag(weights) for _ in range(len(Mlist))]
            elif weights.shape == (Mlist[0].shape[0], Mlist[0].shape[0]):
                # single weight matrix for each sample
                weights = [weights for _ in range(len(Mlist))]
            else:
                raise ValueError("Invalid input for 'weights' argument")

    # calculate Qs
    if weights is None:
        Qlist = [M.T @ M for M in Mlist]
    else:
        assert len(weights) == len(Mlist)
        Qlist = [np.dot(np.dot(M.T, weight), M) for M, weight in zip(Mlist, weights)]

    # output
    if normalize:
        return np.mean(Qlist, axis=0)
    else:
        return np.sum(Qlist, axis=0)


@dataclass
class QCQPPrimalResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    opt_result: Optional[scipy.optimize.OptimizeResult] = None
    value: Optional[float] = None
    x: Optional[np.ndarray] = None


def solve_qcqp(Q: np.ndarray, constraint_funs: List[QuadraticFun], x0: np.ndarray,
               solver_kwargs: Optional[dict] = None) -> QCQPPrimalResult:
    # cost function
    def cost(x):
        return x.T @ Q @ x

    def jac_cost(x):
        return 2.0 * x.T @ Q

    # constraints
    constraints = []
    for fun in constraint_funs:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, f=fun: f.eval(x),
            'jac': lambda x, f=fun: f.jac(x),
        })

    # optimize
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        method='SLSQP',
        options={'maxiter': 1000},
        tol=1e-9,
    )

    # initialize result and start time
    result = QCQPPrimalResult()

    # optimize
    start_time = time.time()
    result.opt_result = scipy.optimize.minimize(
        fun=cost,
        x0=x0,
        jac=jac_cost,
        constraints=constraints,
        **solver_kwargs
    )
    result.run_time = time.time() - start_time

    # store results
    result.success = result.opt_result.success
    if result.opt_result.success:
        result.value = result.opt_result.fun
        result.x = result.opt_result.x
    else:
        result.message = result.opt_result.message

    return result


def is_global(cost: float, gap: float, eps: float = 1e-5):
    gap_rel = gap / cost
    return np.abs(gap) < eps or np.abs(gap_rel) < eps


def verify_qcqp_result_globality(Q: np.ndarray, constraint_funs: List[QuadraticFun], result: QCQPPrimalResult,
                                 eps=1e-6) -> bool:
    # check input
    if not result.success or result.x is None:
        return False

    # solve Z(l) x_sol = 0 for l with Z(l) = Q + l_1 P_1 + ... + l_m P_m
    Qx_vec = Q @ result.x
    Px_vecs = [con.A @ result.x for con in constraint_funs]

    # create and solve LSE
    A = np.column_stack(Px_vecs)
    b = -Qx_vec
    lin_result = solve_linear_problem(A, b)
    if not lin_result.success:
        logger.warning("Could not solve linear problem")
        return False

    # check result value
    if lin_result.value > eps:
        return False

    # check if Z(l) is positive semidefinite
    lambdas = lin_result.x
    Z = Q + np.sum([lam * con.A for lam, con in zip(lambdas, constraint_funs)], axis=0)
    vals = np.linalg.eigvals(Z)
    return np.all(vals > -eps)


@dataclass
class QCQPDualResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    problem: Optional[cvx.Problem] = None
    value: Optional[float] = None
    lambdas: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None


@dataclass
class DualRecoveryResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    value: Optional[float] = None
    x: Optional[np.ndarray] = None
    duality_gap: Optional[float] = None
    is_global: bool = False
    ns_dim: Optional[int] = None
    aux_data: Dict = field(default_factory=dict)


def solve_qcqp_dual(Q: np.ndarray, constraint_funs: List[QuadraticFun],
                    solver_kwargs: Optional[dict] = None) -> QCQPDualResult:
    # initialize output
    result = QCQPDualResult()

    # prepare constraints
    P_list = [fun.A for fun in constraint_funs]
    c_arr = np.array([fun.c for fun in constraint_funs])
    if np.sum(np.abs(c_arr)) == 0.0:
        result.message = "The scalar of at least one constraint must be nonzero"
        return result

    # cost function
    lambdas = cvx.Variable(len(P_list))
    dual_cost = c_arr @ lambdas
    obj = cvx.Maximize(dual_cost)

    # constraints
    P = stack_numpy_to_cvx(np.dstack(P_list), lambdas)
    Z = Q + P
    dual_constraints = [cvx.PSD(Z)]

    # solver args
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        abstol=1e-8,
        max_iters=1000,
        refinement=1
    )

    # dual problem
    problem = cvx.Problem(obj, dual_constraints)
    try:
        # solve
        start_time = time.time()
        problem.solve(
            solver=cvx.CVXOPT,
            kktsolver='robust',
            **solver_kwargs
        )
        result.run_time = time.time() - start_time

        # check solution
        if Z.value is None:
            result.message = f"No dual solution found: problem status is '{problem.status}'"
            return result

    except (ArithmeticError, cvx.error.SolverError, ZeroDivisionError) as e:
        result.message = f"{type(e).__name__}: {str(e)}"
        return result

    # result
    result.success = True
    result.problem = problem
    result.value = problem.value
    result.lambdas = lambdas.value
    result.Z = Z.value
    return result


@dataclass
class QCQPRelaxedResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    problem: Optional[cvx.Problem] = None
    value: Optional[float] = None
    X: Optional[np.ndarray] = None


@dataclass
class SDRRecoveryResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    value: Optional[float] = None
    x: Optional[np.ndarray] = None
    sdr_gap: Optional[float] = None
    is_global: bool = False
    rank: Optional[int] = None
    aux_data: Dict = field(default_factory=dict)


def solve_qcqp_sdr(Q: np.ndarray, constraint_funs: List[QuadraticFun],
                   solver_kwargs: Optional[dict] = None) -> QCQPRelaxedResult:
    # initialize output
    result = QCQPRelaxedResult()

    # prepare input
    x_dim = Q.shape[0]

    # cost function
    X = cvx.Variable((x_dim, x_dim), symmetric=True)
    rel_cost = cvx.trace(Q @ X)
    obj = cvx.Minimize(rel_cost)

    # constraints
    rel_constraints = [cvx.trace(fun.A @ X) + fun.c == 0
                       for fun in constraint_funs]
    rel_constraints.append(cvx.PSD(X))

    # solver args
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        abstol=1e-8,
        max_iters=1000,
        refinement=1
    )

    # relaxed problem
    problem = cvx.Problem(obj, rel_constraints)
    try:
        # solve
        start_time = time.time()
        problem.solve(
            solver=cvx.CVXOPT,
            kktsolver='robust',
            **solver_kwargs
        )
        result.run_time = time.time() - start_time

        # check solution
        if X.value is None:
            result.message = f"No dual solution found: problem status is '{problem.status}'"
            return result

    except (cvx.error.SolverError, ArithmeticError) as e:
        result.message = f"{type(e).__name__}: {str(e)}"
        return result

    # result
    result.success = True
    result.problem = problem
    result.value = problem.value
    result.X = X.value
    return result


class QCQPSolveMethod(Enum):
    NONE = auto()
    QCQP = auto()
    DUAL = auto()
    SDR = auto()


@dataclass
class QCQPOptResult:
    success: bool = False
    message: str = ""
    x: Optional[np.ndarray] = None
    value: Optional[float] = None
    gap: Optional[float] = None
    is_global: bool = False
    method: QCQPSolveMethod = QCQPSolveMethod.NONE
    sdr_result: Optional[QCQPRelaxedResult] = None
    sdr_recovery: Optional[SDRRecoveryResult] = None
    dual_result: Optional[QCQPDualResult] = None
    dual_recovery: Optional[DualRecoveryResult] = None
    primal_result: Optional[QCQPPrimalResult] = None


class _QCQPProblem(metaclass=ABCMeta):
    def __init__(self, Q: np.ndarray, constraint_funs: List[QuadraticFun]):
        self.Q = Q
        self.constraint_funs = constraint_funs

    @abstractmethod
    def recover_from_dual(self, dual_result: QCQPDualResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                          **kwargs) -> DualRecoveryResult:
        raise NotImplementedError

    @abstractmethod
    def recover_from_sdr(self, relaxed_result: QCQPRelaxedResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                         **kwargs) -> SDRRecoveryResult:
        raise NotImplementedError

    def solve_primal(self, x0: np.ndarray, **kwargs) -> QCQPPrimalResult:
        return solve_qcqp(self.Q, self.constraint_funs, x0, **kwargs)

    def solve_dual(self, sol_kwargs: Optional[Dict] = None, rec_kwargs: Optional[Dict] = None) \
            -> Tuple[QCQPDualResult, Optional[DualRecoveryResult]]:
        # default arguments
        if sol_kwargs is None:
            sol_kwargs = {}
        if rec_kwargs is None:
            rec_kwargs = {}

        # solve
        result = solve_qcqp_dual(self.Q, self.constraint_funs, **sol_kwargs)

        # recovery
        if result.success:
            recovery = self.recover_from_dual(result, self.Q, self.constraint_funs, **rec_kwargs)
        else:
            recovery = None

        return result, recovery

    def solve_sdr(self, sol_kwargs: Optional[Dict] = None, rec_kwargs: Optional[Dict] = None) \
            -> Tuple[QCQPRelaxedResult, Optional[SDRRecoveryResult]]:
        # default arguments
        if sol_kwargs is None:
            sol_kwargs = {}
        if rec_kwargs is None:
            rec_kwargs = {}

        # solve
        result = solve_qcqp_sdr(self.Q, self.constraint_funs, **sol_kwargs)

        # recovery
        if result.success:
            recovery = self.recover_from_sdr(result, self.Q, self.constraint_funs, **rec_kwargs)
        else:
            recovery = None

        return result, recovery

    def solve(self, x0: Optional[np.ndarray] = None, force_x0: bool = False,
              use_sdr: bool = True, use_dual: bool = True, use_qcqp: bool = True, qcqp_first: bool = False,
              sdr_kwargs: Optional[Dict] = None, sdr_rec_kwargs: Optional[Dict] = None,
              dual_kwargs: Optional[Dict] = None, dual_rec_kwargs: Optional[Dict] = None,
              qcqp_kwargs: Optional[Dict] = None, qcqp_ver_kwargs: Optional[Dict] = None) -> QCQPOptResult:

        # default arguments
        sdr_kwargs = add_default_kwargs(sdr_kwargs)
        sdr_rec_kwargs = add_default_kwargs(
            sdr_rec_kwargs, assume_rank1=len(self.constraint_funs) <= 2)
        dual_kwargs = add_default_kwargs(dual_kwargs)
        dual_rec_kwargs = add_default_kwargs(
            dual_rec_kwargs, optimize=False)
        qcqp_kwargs = add_default_kwargs(qcqp_kwargs)
        qcqp_ver_kwargs = add_default_kwargs(qcqp_ver_kwargs)

        # initialize result
        res = QCQPOptResult()

        # primal problem first
        if use_qcqp and qcqp_first:
            # initial variable
            if x0 is None:
                logger.warn("Using zero vector as default initial variable")
                x0_first = np.zeros(self.Q.shape[0])
            else:
                x0_first = x0

            # solve primal problem
            res.primal_result = self.solve_primal(x0_first, **qcqp_kwargs)

            # result result
            if res.primal_result.success:
                res.success = True
                res.x = res.primal_result.x
                res.value = res.primal_result.value
                res.method = QCQPSolveMethod.QCQP
                res.is_global = verify_qcqp_result_globality(self.Q, self.constraint_funs, res.primal_result,
                                                             **qcqp_ver_kwargs)

                # stop early if global
                if res.is_global:
                    return res

        # semidefinite relaxation
        if use_sdr and len(self.constraint_funs) <= 2:
            res.sdr_result, res.sdr_recovery = self.solve_sdr(sdr_kwargs, sdr_rec_kwargs)

            if res.sdr_recovery is not None and res.sdr_recovery.success:
                res.success = True
                res.x = res.sdr_recovery.x
                res.value = res.sdr_recovery.value
                res.gap = res.sdr_recovery.sdr_gap
                res.is_global = res.sdr_recovery.is_global
                res.method = QCQPSolveMethod.SDR

                # stop early if global
                if res.is_global:
                    return res

        # dual problem
        if use_dual:
            res.dual_result, res.dual_recovery = self.solve_dual(dual_kwargs, dual_rec_kwargs)

            if res.dual_recovery is not None and res.dual_recovery.success:
                res.success = True
                res.x = res.dual_recovery.x
                res.value = res.dual_recovery.value
                res.gap = res.dual_recovery.duality_gap
                res.is_global = res.dual_recovery.is_global
                res.method = QCQPSolveMethod.DUAL

                # stop early if global
                if res.is_global:
                    return res

        # primal problem last
        if use_qcqp and not qcqp_first:
            # initial variable
            if x0 is None or not force_x0:
                # no x0 given or given x0 not enforced
                if res.dual_recovery is not None and res.dual_recovery.x is not None:
                    x0 = res.dual_recovery.x
                elif res.sdr_recovery is not None and res.sdr_recovery.x is not None:
                    x0 = res.sdr_recovery.x

            # emergency initial solution
            if x0 is None:
                logger.warn("Using zero vector as default initial variable")
                x0 = np.zeros(self.Q.shape[0])

            # solve primal problem
            res.primal_result = self.solve_primal(x0, **qcqp_kwargs)

            # result result
            if res.primal_result.success:
                # calculate duality gap, if dual result was calculated
                gap = None
                if res.dual_result is not None and res.dual_result.success:
                    gap = res.primal_result.value - res.dual_result.value

                # compare to recovered dual result and overwrite if primal result is better
                if res.dual_recovery is None or not res.dual_recovery.success or \
                        (gap is not None and np.abs(gap) < np.abs(res.dual_recovery.duality_gap)):
                    res.success = True
                    res.x = res.primal_result.x
                    res.value = res.primal_result.value
                    res.gap = gap
                    res.method = QCQPSolveMethod.QCQP

                    # globality of primal solution
                    if gap is None:
                        res.is_global = verify_qcqp_result_globality(self.Q, self.constraint_funs, res.primal_result,
                                                                     **qcqp_ver_kwargs)
                    else:
                        res.is_global = is_global(res.primal_result.value, gap)

        # generate message if no success
        if not res.success:
            msgs = []
            if res.primal_result is not None and not res.primal_result.success:
                msgs.append(f"Primal: {res.primal_result.message}")
            if res.sdr_result is not None and not res.sdr_result.success:
                msgs.append(f"SDR Res: {res.sdr_result.message}")
            if res.sdr_recovery is not None and not res.sdr_recovery.success:
                msgs.append(f"SDR Rec: {res.sdr_recovery.message}")
            if res.dual_result is not None and not res.dual_result.success:
                msgs.append(f"Dual Res: {res.dual_result.message}")
            if res.dual_recovery is not None and not res.dual_recovery.success:
                msgs.append(f"Dual Rec: {res.dual_recovery.message}")
            res.message = ', '.join(msgs)

        return res
