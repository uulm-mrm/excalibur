import time

import motion3d as m3d
import numpy as np

from ...base import PairCalibrationResult, TransformPair, MultiTransformPair
from excalibur.optimization.dq import dq_constraints, dq_translation_norm_constraint, solve_qcqp_dq
from excalibur.utils.logging import MessageLevel, Message


# indices
REAL_INDICES = np.array([0, 1, 2, 3])
DUAL_INDICES = np.array([4, 5, 6, 7])


def _solution_to_dq(solution, real_indices, dual_indices):
    if solution[real_indices][0] >= 0.0:
        dq = m3d.DualQuaternionTransform(solution[real_indices], solution[dual_indices], unsafe=True)
    else:
        dq = m3d.DualQuaternionTransform(-solution[real_indices], -solution[dual_indices], unsafe=True)
    dq.normalized_()
    return dq


def optimize(Q, frame_ids, fast=False, t_norms=None, x0=None, **kwargs):
    # check input
    num_dqs = len(frame_ids.x) + len(frame_ids.y) if frame_ids is not None else 2
    dim = num_dqs * 8
    assert Q.shape == (dim, dim)

    # adjust kwargs
    if fast:
        kwargs['use_sdr'] = False
        kwargs['use_dual'] = False
        kwargs['use_qcqp'] = True
        kwargs['qcqp_first'] = True

    # initialize result
    result = PairCalibrationResult()

    # initial solution for primal optimization
    if x0 is None:
        x0 = np.zeros(dim)
        for i in range(num_dqs):
            x0[8 * i] = 1.0

    # constraints
    constraints = []
    for idx in range(num_dqs):
        constraints.extend(dq_constraints(dim, REAL_INDICES + 8 * idx, DUAL_INDICES + 8 * idx))
    if t_norms is not None:
        if frame_ids is None:
            assert isinstance(t_norms, list) and len(t_norms) == 2
            if t_norms[0] is not None:
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES, t_norms[0]))
            if t_norms[1] is not None:
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES + 8, t_norms[1]))
        else:
            assert isinstance(t_norms, dict)
            for idx, frame in enumerate([*frame_ids.x, *frame_ids.y]):
                if frame not in t_norms:
                    continue
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES + 8 * idx, t_norms[frame]))

    # solve
    start_time = time.time()
    qcqp_result = solve_qcqp_dq(Q, constraints, REAL_INDICES, DUAL_INDICES, x0, **kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not qcqp_result.success:
        result.aux_data = {'qcqp_result': qcqp_result}
        result.msgs.append(Message(text=f"Solving failed", level=MessageLevel.FATAL))
        return result

    # construct dual quaternions
    x_est = qcqp_result.x
    if frame_ids is None:
        transform_x = _solution_to_dq(x_est, REAL_INDICES, DUAL_INDICES).normalized_()
        transform_y = _solution_to_dq(x_est, REAL_INDICES + 8, DUAL_INDICES + 8).normalized_()
        result.calib = TransformPair(x=transform_x, y=transform_y)
    else:
        transforms = [_solution_to_dq(x_est, REAL_INDICES + 8 * idx, DUAL_INDICES + 8 * idx)
                      for idx in range(num_dqs)]
        transforms_x = {frame: calib for frame, calib in zip(frame_ids.x, transforms[:len(frame_ids.x)])}
        transforms_y = {frame: calib for frame, calib in zip(frame_ids.y, transforms[len(frame_ids.x):])}
        result.calib = MultiTransformPair(x=transforms_x, y=transforms_y)

    # calculate final cost
    final_cost = x_est.T @ Q @ x_est

    # result
    result.success = True
    result.aux_data = {
        'qcqp_result': qcqp_result,
        'x': x_est,
        'cost': final_cost,
        'gap': qcqp_result.gap,
        'is_global': qcqp_result.is_global,
    }
    return result
