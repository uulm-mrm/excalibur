import time

import motion3d as m3d
import numpy as np

from excalibur.optimization.dq import dq_constraints, dq_translation_norm_constraint, solve_qcqp_dq
from excalibur.utils.logging import MessageLevel, Message
from ...base import CalibrationResult


# constraint matrices
DIM = 8
REAL_INDICES = [0, 1, 2, 3]
DUAL_INDICES = [4, 5, 6, 7]


def optimize(Q, fast=False, t_norm=None, x0=None, **kwargs):
    # check input
    assert Q.shape == (DIM, DIM)

    # adjust kwargs
    if fast:
        kwargs['use_sdr'] = False
        kwargs['use_dual'] = False
        kwargs['use_qcqp'] = True
        kwargs['qcqp_first'] = True

    # initialize result
    result = CalibrationResult()

    # initial solution for primal optimization
    if x0 is None:
        x0 = np.zeros(DIM)
        x0[REAL_INDICES[0]] = 1.0

    # constraints
    constraints = [
        *dq_constraints(DIM, REAL_INDICES, DUAL_INDICES),
    ]
    if t_norm:
        constraints.append(dq_translation_norm_constraint(DIM, DUAL_INDICES, t_norm))

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

    # result
    result.success = True
    result.calib = dual_quat_solution
    result.aux_data = {
        'qcqp_result': qcqp_result,
        'is_global': qcqp_result.is_global,
    }
    return result
