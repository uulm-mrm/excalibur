import time

import motion3d as m3d
import numpy as np

from excalibur.optimization.dq import calculate_nullspace_factors_daniilidis
from ...base import CalibrationResult


REAL_INDICES = [0, 1, 2, 3]
DUAL_INDICES = [4, 5, 6, 7]


def solve_svd(Mlist):
    # prepare result and matrix
    result = CalibrationResult()
    T = np.concatenate(Mlist, axis=0).T

    # start time
    start_time = time.time()

    # solve using svd
    U, S, Vh = np.linalg.svd(T)

    # retrieve solution from nullspace
    lambdas = calculate_nullspace_factors_daniilidis(U[:, -2:], REAL_INDICES, DUAL_INDICES)
    solution = U[:, -2:] @ lambdas

    # store time
    result.run_time = time.time() - start_time

    # create dual quaternion
    dual_quat_solution = m3d.DualQuaternionTransform(solution, unsafe=True)
    dual_quat_solution = dual_quat_solution.normalized()

    result.success = True
    result.calib = dual_quat_solution
    result.aux_data = {
        'x': solution
    }
    return result
