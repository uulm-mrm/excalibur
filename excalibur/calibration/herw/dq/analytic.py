import time

import motion3d as m3d
import numpy as np

from ...base import CalibrationResult, TransformPair

from excalibur.optimization.dq import calculate_nullspace_factors_daniilidis


REAL_INDICES = [0, 1, 2, 3]
DUAL_INDICES = [4, 5, 6, 7]


def solve_svd(Mlist):
    # initialize result
    result = CalibrationResult()

    # prepare matrix
    T = np.concatenate(Mlist, axis=0).T

    # start time measurement
    start_time = time.time()

    # solve using svd
    U, S, Vh = np.linalg.svd(T)

    # retrieve solution from nullspace
    lambdas = calculate_nullspace_factors_daniilidis(U[:, -2:], REAL_INDICES, DUAL_INDICES)
    if lambdas is None or np.isnan(lambdas[0]) or np.isnan(lambdas[1]):
        result.run_time = time.time() - start_time
        return result
    x_est = U[:, -2:] @ lambdas

    # stop time
    result.run_time = time.time() - start_time

    # create dual quaternion
    dual_quat_solution_x = m3d.DualQuaternionTransform(x_est[:8], unsafe=True).normalized_()
    dual_quat_solution_y = m3d.DualQuaternionTransform(x_est[8:], unsafe=True).normalized_()

    # calculate final cost
    final_cost = np.linalg.norm(T.T @ x_est)

    # result
    result.success = True
    result.calib = TransformPair(x=dual_quat_solution_x, y=dual_quat_solution_y)
    result.aux_data = {
        'x': x_est,
        'cost': final_cost,
    }
    return result
