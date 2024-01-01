from dataclasses import dataclass

import motion3d as m3d
import numpy as np

from excalibur.optimization.linear import solve_linear_problem
from excalibur.utils.math import sorted_eig


_SAMPLES = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
])
_SAMPLES = _SAMPLES / np.linalg.norm(_SAMPLES, axis=1, keepdims=True)


def _get_condition_number(Q, calib, samples, delta):
    # calculate costs
    trafo_vecs = np.array([s.asType(m3d.TransformType.kDualQuaternion).toArray() for s in [calib, *samples]])
    trafo_costs = np.array([trafo_vecs[row, :] @ Q @ trafo_vecs[row, :].T for row in range(trafo_vecs.shape[0])])
    cost_diffs = trafo_costs[1:] - trafo_costs[0]
    if np.min(cost_diffs) <= -1e-4:
        # warnings.warn(f"Solution might not be globally optimal, "
        #               f"skipping conditioning (cost diff {np.min(cost_diffs)}).")
        return None, None

    # get S matrix
    p_mat = (delta ** 2) * np.column_stack((
        _SAMPLES ** 2,
        2 * _SAMPLES[:, 0] * _SAMPLES[:, 1],
        2 * _SAMPLES[:, 0] * _SAMPLES[:, 2],
        2 * _SAMPLES[:, 1] * _SAMPLES[:, 2]
    ))
    result = solve_linear_problem(p_mat, cost_diffs)
    if not result.success:
        return None, None

    s_vec = result.x
    S_mat = np.array([
        [s_vec[0], s_vec[3], s_vec[4]],
        [s_vec[3], s_vec[1], s_vec[5]],
        [s_vec[4], s_vec[5], s_vec[2]],
    ])

    # check result
    for row in range(_SAMPLES.shape[0]):
        pred = delta**2 * _SAMPLES[row, :] @ S_mat @ _SAMPLES[row, :]
        assert np.abs(pred - cost_diffs[row]) < 1e-6

    # eigenvalue decomposition
    vals, vecs = sorted_eig(S_mat)
    cond = vals[-1] / vals[0]
    cond_vec = vecs[:, 0]
    return cond, cond_vec


@dataclass
class ConditioningData:
    trans_cond: float
    trans_vec: np.ndarray
    rot_cond: float
    rot_vec: np.ndarray


def get_conditioning(Q, calib, delta_t=0.1, delta_r=np.deg2rad(0.1)):
    # create samples
    trans_samples = [
        m3d.AxisAngleTransform(delta_t * sample, 0.0, [1.0, 0.0, 0.0]) * calib
        for sample in _SAMPLES
    ]
    rot_samples = [
        m3d.AxisAngleTransform(np.zeros(3), delta_r, sample) * calib
        for sample in _SAMPLES
    ]

    # conditioning
    trans_cond, trans_vec = _get_condition_number(Q, calib, trans_samples, delta_t)
    rot_cond, rot_vec = _get_condition_number(Q, calib, rot_samples, delta_r)
    return ConditioningData(
        trans_cond=trans_cond,
        trans_vec=trans_vec,
        rot_cond=rot_cond,
        rot_vec=rot_vec,
    )
