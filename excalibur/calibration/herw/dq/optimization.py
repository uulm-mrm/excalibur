import time

import motion3d as m3d
import numpy as np

from ...base import PairCalibrationResult, TransformPair, MultiTransformPair
from excalibur.optimization.dq import dq_constraints, dq_translation_norm_constraint, QCQPProblemDQ
from excalibur.optimization.utils import MultiThreshold


# indices
REAL_INDICES = np.array([0, 1, 2, 3])
DUAL_INDICES = np.array([4, 5, 6, 7])

DEFAULT_DUAL_EPS = 1e-3
DEFAULT_T_NORM_EPS = 1e-3


def _solution_to_dq(solution, real_indices, dual_indices):
    if solution[real_indices][0] >= 0.0:
        dq = m3d.DualQuaternionTransform(solution[real_indices], solution[dual_indices], unsafe=True)
    else:
        dq = m3d.DualQuaternionTransform(-solution[real_indices], -solution[dual_indices], unsafe=True)
    dq.normalized_()
    return dq


def optimize(Q, frame_ids, fast=False, t_norms=None, x0=None, dual_eps=DEFAULT_DUAL_EPS,
             t_norm_eps=DEFAULT_T_NORM_EPS, **kwargs):
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
        if t_norms is not None:
            raise NotImplementedError("Initialization with t_norms is not implemented")
        x0 = np.zeros(dim)
        for i in range(num_dqs):
            x0[8 * i] = 1.0

    # initialize eps
    eps_constraints = None
    if dual_eps is not None or t_norm_eps is not None:
        eps_constraints = MultiThreshold()

    # constraints
    constraints = []
    for idx in range(num_dqs):
        dq_con = dq_constraints(dim, REAL_INDICES + 8 * idx, DUAL_INDICES + 8 * idx)
        constraints.extend(dq_con)
        if dual_eps is not None:
            eps_constraints.set(len(constraints) - 1, dual_eps)
    if t_norms is not None:
        if frame_ids is None:
            assert isinstance(t_norms, list) and len(t_norms) == 2
            if t_norms[0] is not None:
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES, t_norms[0]))
                if t_norm_eps is not None:
                    eps_constraints.set(len(constraints) - 1, t_norm_eps)
            if t_norms[1] is not None:
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES + 8, t_norms[1]))
                if t_norm_eps is not None:
                    eps_constraints.set(len(constraints) - 1, t_norm_eps)
        else:
            assert isinstance(t_norms, dict)
            for idx, frame in enumerate([*frame_ids.x, *frame_ids.y]):
                if frame not in t_norms:
                    continue
                constraints.append(dq_translation_norm_constraint(dim, DUAL_INDICES + 8 * idx, t_norms[frame]))
                if t_norm_eps is not None:
                    eps_constraints.set(len(constraints) - 1, t_norm_eps)

    # set eps
    if eps_constraints is not None:
        if 'sdr_rec_kwargs' not in kwargs:
            kwargs['sdr_rec_kwargs'] = {'eps_constraints': eps_constraints}
        if 'dual_rec_kwargs' not in kwargs:
            kwargs['dual_rec_kwargs'] = {'eps_constraints': eps_constraints}

    # force initial x0 for multi calibration
    if num_dqs > 2 and 'force_x0' not in kwargs:
        kwargs['force_x0'] = True

    # solve
    start_time = time.time()
    problem = QCQPProblemDQ(Q, constraints, REAL_INDICES, DUAL_INDICES)
    qcqp_result = problem.solve(x0=x0, **kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not qcqp_result.success:
        result.message = qcqp_result.message
        result.aux_data = {'qcqp_result': qcqp_result}
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
