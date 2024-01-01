import time

import motion3d as m3d
import numpy as np
import scipy.optimize

from .generation import SchmidtData
from ...base import CalibrationResult, CalibrationResultScaled
from excalibur.optimization.hm import rotmat_constraints_hom, QCQPProblemHM
from excalibur.utils.math import schur_complement_A
from excalibur.utils.parameters import add_default_kwargs


# indices
DIM = 13
HOM_INDEX_RED = -1

# constraints
RM_CONSTRAINTS = rotmat_constraints_hom()


def optimize_qcqp(Q, solver_kwargs=None, recovery_kwargs=None):
    # check input
    assert Q.shape == (DIM, DIM)

    # default arguments
    solver_kwargs = add_default_kwargs(solver_kwargs)
    recovery_kwargs = add_default_kwargs(recovery_kwargs)

    # initialize result
    result = CalibrationResult()

    # start time measurement
    start_time = time.time()

    # reduce Q
    try:
        Q_red = schur_complement_A(Q, 3, 3)
    except np.linalg.LinAlgError:
        result.message = "Singular matrix in Schur complement"
        return result

    # solve dual problem
    problem = QCQPProblemHM(Q_red, RM_CONSTRAINTS, HOM_INDEX_RED)
    dual_result, dual_recovery = problem.solve_dual(solver_kwargs, recovery_kwargs)

    # check success
    if not dual_result.success:
        result.message = f"Optimization failed ({dual_result.message})"
        return result

    if dual_recovery is None or not dual_recovery.success:
        result.message = "Recovery failed"
        if dual_recovery is not None:
            result.message += f" ({dual_recovery.message})"
        return result

    # construct transformation
    x_est = dual_recovery.x
    x_rotmat = x_est[:9].reshape(3, 3).T
    x_trans = - np.linalg.inv(Q[:3, :3]) @ Q[:3, 3:] @ x_est

    # stop time
    result.run_time = time.time() - start_time

    matrix_solution = m3d.MatrixTransform(x_trans, x_rotmat, unsafe=True)
    matrix_solution.normalized_().inverse_()  # Giamou et al. is defined the other way round

    # result
    result.success = True
    result.calib = matrix_solution
    result.aux_data = {
        'dual_result': dual_result,
        'dual_recovery': dual_recovery,
        'is_global': dual_recovery.is_global,
        'gap': dual_recovery.duality_gap,
    }
    return result


def optimize_qcqp_scaled(Q, solver_kwargs=None, recovery_kwargs=None):
    # check input
    assert Q.shape == (DIM, DIM)

    # default arguments
    solver_kwargs = add_default_kwargs(solver_kwargs)
    recovery_kwargs = add_default_kwargs(recovery_kwargs)

    # initialize result
    result = CalibrationResultScaled()

    # start time measurement
    start_time = time.time()

    # reduce Q
    Q_red_ext = np.zeros((10, 10))
    Q_red_ext[:9, :9] = schur_complement_A(Q, 4, 4)

    # solve dual problem
    problem = QCQPProblemHM(Q_red_ext, RM_CONSTRAINTS, HOM_INDEX_RED)
    dual_result, dual_recovery = problem.solve_dual(solver_kwargs, recovery_kwargs)

    # check success
    if not dual_result.success:
        result.message = f"Optimization failed ({dual_result.message})"
        return result

    if dual_recovery is None or not dual_recovery.success:
        result.message = "Recovery failed"
        if dual_recovery is not None:
            result.message += f" ({dual_recovery.message})"
        return result

    # construct transformation
    x_est = dual_recovery.x
    x_rotmat = x_est[:9].reshape(3, 3).T
    x_trans_alpha = - np.linalg.inv(Q[:4, :4]) @ Q[:4, 4:] @ x_est[:9]
    x_trans = x_trans_alpha[:3]
    alpha = x_trans_alpha[-1]

    # stop time
    result.run_time = time.time() - start_time

    matrix_solution = m3d.MatrixTransform(x_trans, x_rotmat, unsafe=True)
    matrix_solution.normalized_().inverse_()  # Wise et al. is defined the other way round

    # result
    result.success = True
    result.calib = matrix_solution
    result.scale = alpha
    result.aux_data = {
        'dual_result': dual_result,
        'dual_recovery': dual_recovery,
        'is_global': dual_recovery.is_global,
        'gap': dual_recovery.duality_gap,
    }
    return result


I33 = np.eye(3)


def _schmidt_costs(x, data: SchmidtData, lam):
    # split x
    x_q = m3d.Quaternion.FromArray(x[:4])
    x_qc = x_q.conjugate()
    x_t = x[4:7]
    x_s = x[7]

    # rotation costs
    part1 = np.sum([(e.getQuaternion() - x_q * h.getQuaternion() * x_qc).squaredNorm()
                    for h, e in zip(data.hand_quat, data.eye_quat)])

    # translation costs
    part2 = np.sum([(
        m3d.Quaternion(0.0, *((I33 - e.getRotationMatrix()) @ x_t - e.getTranslation())) +
        x_q * m3d.Quaternion(0.0, *(x_s * h.getTranslation())) * x_qc
    ).squaredNorm() for h, e in zip(data.hand_hm, data.eye_hm)])

    # constraint costs
    part3 = np.square(1.0 - x_q.squaredNorm())

    return part1 + part2 + lam * part3


def _schmidt_costs_improved(x, data: SchmidtData, lam):
    # split x
    x_q_vec = x[:4]
    x_q = m3d.Quaternion.FromArray(x[:4])
    x_qc = x_q.conjugate()
    x_t = x[4:7]
    x_s = x[7]

    # matrices
    x_qc_neg = x_qc.toNegativeMatrix()

    # rotation costs
    # xq * hq
    xh_quat = data.h_quat_neg @ x_q_vec

    # xq * hq * xq'
    xhx_quat_vec = (x_qc_neg @ xh_quat.T).T.flatten()

    # eq - xq * hq * xq'
    part1_diff = data.e_quat_vec - xhx_quat_vec
    part1 = part1_diff.dot(part1_diff)

    # translation costs
    # Quat(0.0, (I33 - eR) * xt - et)
    trans_diff = data.e_rot_diff @ x_t - data.e_tran_vec
    trans_diff_quat = np.column_stack((np.zeros(trans_diff.shape[0]), trans_diff))
    trans_diff_quat_vec = trans_diff_quat.flatten()

    # xq * Quat(0.0, xs * ht)
    h_tran_quat_scaled = x_s * data.h_tran_quat_neg
    xht_quat_vec = h_tran_quat_scaled @ x_q_vec
    xht_quat = xht_quat_vec.reshape(int(len(xht_quat_vec) / 4), 4)

    # xq * Quat(0.0, xs * ht) * xq'
    xhtx_quat_vec = (x_qc_neg @ xht_quat.T).T.flatten()

    # add translation cost parts
    part2_sum = trans_diff_quat_vec + xhtx_quat_vec
    part2 = part2_sum.dot(part2_sum)

    # constraint costs
    part3 = np.square(1.0 - x_q.squaredNorm())

    return part1 + part2 + lam * part3


def optimize_schmidt(data: SchmidtData, x0=None, lam=2e-6, improved=True, solver_kwargs=None):
    # check input
    assert len(data.hand_hm) == len(data.hand_quat) == len(data.eye_hm) == len(data.eye_quat)

    # initialize result
    result = CalibrationResultScaled()

    # initial solution
    if x0 is None:
        x0 = np.zeros(8)
        x0[0] = 1.0
        x0[-1] = 1.0

    # solver arguments
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        method='SLSQP',
        tol=1e-9
    )

    # select cost function
    if improved:
        def cost_fun(x):
            return _schmidt_costs_improved(x, data, lam)
    else:
        def cost_fun(x):
            return _schmidt_costs(x, data, lam)

    # optimize
    start_time = time.time()
    opt_result = scipy.optimize.minimize(
        cost_fun,
        x0,
        **solver_kwargs
    )
    if not opt_result.success:
        result.message = f"Solving failed ({opt_result.message})"
        result.opt_result = {'opt_result': opt_result}
        return result

    # recover solution
    res_q = opt_result.x[:4]
    if res_q[0] < 0:
        res_q *= -1
    res_t = opt_result.x[4:7]
    res_s = opt_result.x[7]

    result.run_time = time.time() - start_time

    # construct quaternion transform
    quat_solution = m3d.QuaternionTransform(res_t, res_q, unsafe=True)
    quat_solution.normalized_()

    # result
    result.success = True
    result.calib = quat_solution
    result.scale = res_s
    result.aux_data = {
        'opt_result': opt_result,
    }
    return result
