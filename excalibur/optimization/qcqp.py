from dataclasses import dataclass
from typing import List, Optional, Union

import cvxpy as cvx
import numpy as np
import scipy.optimize
import time

from excalibur.utils.parameters import add_default_kwargs
from .cvx_utils import stack_numpy_to_cvx
from .linear import solve_linear_problem
from excalibur.utils.logging import logger


@dataclass
class QuadraticFun:
    A: np.ndarray
    c: float = 0.0

    def eval(self, x: np.ndarray):
        return x.T @ self.A @ x + self.c

    def jac(self, x: np.ndarray):
        return 2 * x.T @ self.A


def generate_quadratic_cost_matrix(Mlist: List[np.ndarray], weights: Optional[Union[np.ndarray, List]] = None,
                                   normalize: bool = False) -> np.ndarray:
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
class QCQPResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    opt_result: Optional[scipy.optimize.OptimizeResult] = None
    value: Optional[float] = None
    x: Optional[np.ndarray] = None


def solve_qcqp(Q: np.ndarray, constraint_funs: List[QuadraticFun], x0: np.ndarray,
               solver_kwargs: Optional[dict] = None) -> QCQPResult:
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
    result = QCQPResult()

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


def verify_qcqp_result_globality(Q: np.ndarray, constraint_funs: List[QuadraticFun], result: QCQPResult,
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
    Z = Q + np.sum([l * con.A for l, con in zip(lambdas, constraint_funs)], axis=0)
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


def solve_qcqp_dual(Q: np.ndarray, constraint_funs: List[QuadraticFun],
                    solver_kwargs: Optional[dict] = None) -> QCQPDualResult:
    # initialize output
    result = QCQPDualResult()

    # prepare constraints
    P_list = [fun.A for fun in constraint_funs]
    c_arr = np.array([fun.c for fun in constraint_funs])
    if np.sum(np.abs(c_arr)) == 0.0:
        logger.fatal("The scalar of at least one constraint must be nonzero")
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
            logger.warning(result.message)
            return result

    except (cvx.error.SolverError, ZeroDivisionError) as e:
        result.message = f"{type(e).__name__}: {str(e)}"
        logger.warning(result.message)
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
            logger.warning(result.message)
            return result

    except (cvx.error.SolverError, ArithmeticError) as e:
        result.message = f"{type(e).__name__}: {str(e)}"
        logger.warning(result.message)
        return result

    # result
    result.success = True
    result.problem = problem
    result.value = problem.value
    result.X = X.value
    return result
