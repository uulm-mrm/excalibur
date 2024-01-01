import time

import motion3d as m3d

from ...base import CalibrationResult
from excalibur.optimization.hm import normalize_rotation_matrix
from excalibur.optimization.qcqp import solve_linear_problem


def solve_linear(A, b):
    # initialize result
    result = CalibrationResult()

    # optimize matrix
    start_time = time.time()
    opt_result = solve_linear_problem(A, b)
    result.run_time = time.time() - start_time

    # check solution
    if not opt_result.success:
        result.message = "No solution found"
        return result

    # extract R and t
    x = opt_result.x
    R = normalize_rotation_matrix(x[:9].reshape(3, 3))
    t = x[9:]

    # create matrix
    if R is None:
        return result
    matrix_solution = m3d.MatrixTransform(t, R, unsafe=True)
    matrix_solution = matrix_solution.normalized()

    # create result
    result.success = True
    result.calib = matrix_solution
    result.aux_data = {
        'x': x
    }
    return result
