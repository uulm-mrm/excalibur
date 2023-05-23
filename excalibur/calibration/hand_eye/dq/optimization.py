import time

import cvxpy as cvx
import motion3d as m3d
import numpy as np
import scipy.optimize

from .generation import SchmidtData, WeiData
from ...base import CalibrationResult, CalibrationResultScaled
from excalibur.optimization.constraints import parallel_constraints_4d
from excalibur.optimization.dq import dq_constraints, dq_real_constraint, dq_recover_from_planar, dq_reduce_to_planar,\
    dq_translation_norm_constraint, solve_qcqp_dq
from excalibur.utils.logging import MessageLevel, Message
from excalibur.utils.parameters import add_default_kwargs


# constraint matrices
REAL_INDICES = np.array([0, 1, 2, 3])
DUAL_INDICES = np.array([4, 5, 6, 7])
REAL_INDICES_SCALED = np.array([8, 9, 10, 11])


def optimize_qcqp(Q, fast=False, planar_only=False, t_norm=None, x0=None, **kwargs):
    # initialize result
    result = CalibrationResult()

    # check input
    assert Q.shape == (8, 8)
    if planar_only and t_norm is not None:
        result.msgs.append(Message(text=f"Planar only and translation norm constraints cannot be combined",
                                   level=MessageLevel.FATAL))
        return result

    # adjust kwargs
    if fast:
        kwargs['use_sdr'] = False
        kwargs['use_dual'] = False
        kwargs['use_qcqp'] = True
        kwargs['qcqp_first'] = True

    # planar only reduction
    if planar_only:
        dim = 4
        real_indices = REAL_INDICES
        dual_indices = DUAL_INDICES
        Q = dq_reduce_to_planar(Q)
    else:
        dim = 8
        real_indices = REAL_INDICES
        dual_indices = DUAL_INDICES

    # initial solution for primal optimization
    if x0 is None:
        x0 = np.zeros(dim)
        x0[real_indices[0]] = 1.0

    # constraints
    if planar_only:
        constraints = [dq_real_constraint(dim, real_indices)]
    else:
        constraints = [*dq_constraints(dim, real_indices, dual_indices)]
    if t_norm:
        constraints.append(dq_translation_norm_constraint(dim, dual_indices, t_norm))

    # solve
    start_time = time.time()
    qcqp_result = solve_qcqp_dq(Q, constraints, real_indices, dual_indices, x0, **kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not qcqp_result.success:
        result.msgs.append(Message(text=f"Solving failed", level=MessageLevel.FATAL))
        return result

    # planar only recovery
    x_est = qcqp_result.x
    if planar_only:
        x_est = dq_recover_from_planar(x_est)

    # construct dual quaternion
    dual_quat_solution = m3d.DualQuaternionTransform(x_est, unsafe=True)
    dual_quat_solution.normalized_()

    # result
    result.success = True
    result.calib = dual_quat_solution
    result.aux_data = {
        'qcqp_result': qcqp_result,
        'is_global': qcqp_result.is_global,
    }
    return result


def optimize_qcqp_scaled(Q, fast=False, reduced=False, t_norm=None, x0=None, **kwargs):
    # check input
    dim = Q.shape[0]
    assert Q.shape == (dim, dim)
    assert dim % 4 == 0
    num_scalings = int(dim / 4) - 2
    assert num_scalings >= 1

    # adjust kwargs
    if fast:
        kwargs['use_sdr'] = False
        kwargs['use_dual'] = False
        kwargs['use_qcqp'] = True
        kwargs['qcqp_first'] = True
        reduced = True

    # initialize result
    result = CalibrationResultScaled()

    # initial solution for primal optimization
    if x0 is None:
        x0 = np.zeros(dim)
        x0[REAL_INDICES[0]] = 1.0
        for s in range(num_scalings):
            x0[REAL_INDICES_SCALED[0] + 4 * s] = 1.0

    # constraints
    constraints = [*dq_constraints(dim, REAL_INDICES, DUAL_INDICES)]
    for s in range(num_scalings):
        scaled_indices = REAL_INDICES_SCALED + 4 * s
        constraints.extend(parallel_constraints_4d(dim, REAL_INDICES, scaled_indices, reduced=reduced))
    if t_norm:
        constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES, t_norm))

    # solve
    start_time = time.time()
    qcqp_result = solve_qcqp_dq(Q, constraints, REAL_INDICES, DUAL_INDICES, x0, **kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not qcqp_result.success:
        result.aux_data = {'qcqp_result': qcqp_result}
        result.msgs.append(Message(text=f"Solving failed", level=MessageLevel.FATAL))
        return result

    # construct dual quaternion
    x_est = qcqp_result.x
    dual_quat_solution = m3d.DualQuaternionTransform(x_est[[*REAL_INDICES, *DUAL_INDICES]], unsafe=True)
    dual_quat_solution.normalized_()

    # get scalings
    if num_scalings == 1:
        scale = x_est[REAL_INDICES].T @ x_est[REAL_INDICES_SCALED]
    else:
        scale = [x_est[REAL_INDICES].T @ x_est[REAL_INDICES_SCALED + 4 * s]
                 for s in range(num_scalings)]

    # result
    result.success = True
    result.calib = dual_quat_solution
    result.scale = scale
    result.aux_data = {
        'qcqp_result': qcqp_result,
        'is_global': qcqp_result.is_global,
    }
    return result


def _schmidt_costs(x, data, lam):
    # split x
    x_real = m3d.Quaternion.FromArray(x[:4])
    x_dual = m3d.Quaternion.FromArray(x[4:])

    # real costs
    part1 = np.sum([(
        e.getReal() * x_real - x_real * h.getReal()
    ).squaredNorm() for h, e in zip(data.hand, data.eye)])

    # dual costs
    sinv = 1.0 / x_real.norm()
    part2 = np.sum([(
        e.getReal() * x_dual + sinv * e.getDual() * x_real -
        x_real * h.getDual() - x_dual * h.getReal()
    ).squaredNorm() for h, e in zip(data.hand, data.eye)])

    # constraint costs
    part3 = 4 * np.square(x_real.toArray().dot(x_dual.toArray()))

    return part1 + part2 + lam * part3


def _schmidt_costs_improved(x, data, lam):
    # split x
    x_real = x[:4]
    x_dual = x[4:]

    # real costs
    mat1 = data.e_real_pos - data.h_real_neg
    part1 = x_real.T @ mat1.T @ mat1 @ x_real

    # dual costs
    sinv = 1.0 / np.sqrt(x_real.dot(x_real))
    vec2 = data.e_real_pos @ x_dual + sinv * data.e_dual_pos @ x_real - \
           data.h_dual_neg @ x_real - data.h_real_neg @ x_dual
    part2 = vec2.T @ vec2

    # constraint costs
    part3 = 4 * np.square(x_real.dot(x_dual))

    return part1 + part2 + lam * part3


def optimize_schmidt(data: SchmidtData, x0=None, lam=2e-6, improved=True, solver_kwargs=None):
    # check input
    assert len(data.hand) == len(data.eye)

    # initialize result
    result = CalibrationResultScaled()

    # initial solution
    if x0 is None:
        x0 = np.zeros(8)
        x0[0] = 1.0

    # solver arguments
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        method='SLSQP',
        tol=1e-9
    )

    # select cost function
    if improved:
        cost_fun = lambda x: _schmidt_costs_improved(x, data, lam)
    else:
        cost_fun = lambda x: _schmidt_costs(x, data, lam)

    # optimize
    start_time = time.time()
    opt_result = scipy.optimize.minimize(
        cost_fun,
        x0,
        **solver_kwargs
    )
    if not opt_result.success:
        result.opt_result = {'opt_result': opt_result}
        result.msgs.append(Message(text=f"Solving failed", level=MessageLevel.FATAL))
        return result

    # recover solution
    res_nd = opt_result.x[:4]
    res_d = opt_result.x[4:]
    res_s = np.sqrt(res_nd.dot(res_nd))
    res_nd /= res_s
    if res_nd[0] < 0:
        res_nd *= -1
    result.run_time = time.time() - start_time

    # construct dual quaternion
    dual_quat_solution = m3d.DualQuaternionTransform(res_nd, res_d, unsafe=True)
    dual_quat_solution.normalized_()

    # result
    result.success = True
    result.calib = dual_quat_solution
    result.scale = res_s
    result.aux_data = {
        'opt_result': opt_result,
    }
    return result


def optimize_wei(data: WeiData, solver_kwargs=None):
    # initialize result
    result = CalibrationResultScaled()

    # solver arguments
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        abstol=1e-8,
    )

    # first solving with SVD for real part
    start_time = time.time()
    _, _, vh = np.linalg.svd(data.V, full_matrices=False)

    r = np.expand_dims(vh[3, :], axis=1)
    if r[0] < 0:
        r *= -1

    # adjust data with r
    C = np.hstack((data.Cs[:, :4], data.Cs[:, 4:] @ r))
    d = data.ds @ r

    # second optimization with SOCP for dual part
    sigma = cvx.Variable((1, 1))
    x = cvx.Variable((5, 1))

    cons = [cvx.norm(C @ x - d, 2) <= sigma]
    cons += [r.T @ x[:4] == 0]

    obj = cvx.Minimize(sigma)
    problem = cvx.Problem(obj, cons)

    # solve SOCP problem
    try:
        # solve optimization problem
        problem.solve(
            **solver_kwargs
        )
    except (cvx.error.SolverError, ZeroDivisionError) as e:
        result.msgs.append(Message(text=f"Solving failed", level=MessageLevel.FATAL))
        return result

    # store run time
    result.run_time = time.time() - start_time

    # recover solution
    d = x.value[:4, 0].squeeze()
    dual_quat_solution = m3d.DualQuaternionTransform(r, d, unsafe=True).normalized_()
    scale = x.value[4, 0]

    # result
    result.success = True
    result.calib = dual_quat_solution
    result.scale = scale
    return result
