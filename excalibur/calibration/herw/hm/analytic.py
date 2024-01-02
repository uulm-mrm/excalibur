import time

import motion3d as m3d
import numpy as np
import scipy

from ...base import PairCalibrationResult, TransformPair, MultiTransformPair
from excalibur.optimization.hm import normalize_rotation_matrix
from excalibur.optimization.linear import solve_linear_problem
from excalibur.utils.math import sorted_eig


def solve_linear(A, b):
    # initialize result
    result = PairCalibrationResult()

    # optimize matrix
    start_time = time.time()
    lin_result = solve_linear_problem(A, b)
    result.run_time = time.time() - start_time

    # check solution
    if not lin_result.success:
        result.message = "No solution found"
        return result

    # extract R and t
    x_est = lin_result.x
    Rx = normalize_rotation_matrix(x_est[:9].reshape(3, 3))
    Ry = normalize_rotation_matrix(x_est[9:18].reshape(3, 3))
    tx = x_est[18:21]
    ty = x_est[21:]

    # create matrices
    matrix_solution_x = m3d.MatrixTransform(tx, Rx, unsafe=True).normalized_()
    matrix_solution_y = m3d.MatrixTransform(ty, Ry, unsafe=True).normalized_()

    # create result
    result.success = True
    result.calib = TransformPair(x=matrix_solution_x, y=matrix_solution_y)
    result.aux_data = {
        'x': x_est
    }
    return result


def solve_shah(T, A, b_data):
    # initialize result
    result = PairCalibrationResult()

    # start timer
    start_time = time.time()

    # rotations
    u, _, vt = np.linalg.svd(T, full_matrices=False)
    Rx = normalize_rotation_matrix(vt[0, :].reshape(3, 3)).T  # transpose since row was taken instead of column
    Ry = normalize_rotation_matrix(u[:, 0].reshape(3, 3)).T

    # prepare b
    blist = [b1 - b2 @ Ry.T.reshape(9, 1) for b1, b2 in b_data]
    b = np.vstack(blist)

    # translations
    t_result = solve_linear_problem(A, b)

    # stop timer
    result.run_time = time.time() - start_time

    # check result
    if not t_result.success:
        result.message = "No solution found for the translation"
        return result

    # extract t
    tx = t_result.x[:3]
    ty = t_result.x[3:]

    # create matrices
    matrix_solution_x = m3d.MatrixTransform(tx, Rx, unsafe=True).normalized_()
    matrix_solution_y = m3d.MatrixTransform(ty, Ry, unsafe=True).normalized_()

    # create result
    result.success = True
    result.calib = TransformPair(x=matrix_solution_x, y=matrix_solution_y)
    result.aux_data = {
        't_result': t_result,
    }
    return result


def solve_wang(data, frame_ids):
    # initialize result
    result = PairCalibrationResult()

    # number of transforms in data
    x_count = int(data.M_rot.shape[0] / 9) - 1

    # start time measurment
    start_time = time.time()

    # solve rotation matrices
    vals, vecs = sorted_eig(data.M_rot)
    r_vec_y = vecs[:9, 0]
    r_vec_x_list = [vecs[(x_id + 1) * 9:(x_id + 2) * 9, 0] for x_id in range(x_count)]

    # project rotation matrices
    Rx_list = [normalize_rotation_matrix(r_vec_x.reshape(3, 3).T) for r_vec_x in r_vec_x_list]
    Ry = normalize_rotation_matrix(r_vec_y.reshape(3, 3).T)

    # create translation matrices
    t_trans = data.t_trans_a - (Ry @ data.t_trans_b).T.reshape(data.t_trans_a.shape)

    # solve translation
    t_vec = scipy.linalg.solve(data.M_trans.T @ data.M_trans, data.M_trans.T @ t_trans)
    t_vec_y = t_vec[:3]
    t_vec_x_list = [t_vec[(s + 1) * 3:(s + 2) * 3, 0] for s in range(x_count)]

    # stop timer
    result.run_time = time.time() - start_time

    # create matrices
    if data.swap:
        matrix_solutions_y = [m3d.MatrixTransform(t, R, unsafe=True).normalized_().inverse_()
                              for t, R in zip(t_vec_x_list, Rx_list)]
        matrix_solutions_x = [m3d.MatrixTransform(t_vec_y, Ry, unsafe=True).normalized_().inverse_()]
    else:
        matrix_solutions_x = [m3d.MatrixTransform(t, R, unsafe=True).normalized_()
                              for t, R in zip(t_vec_x_list, Rx_list)]
        matrix_solutions_y = [m3d.MatrixTransform(t_vec_y, Ry, unsafe=True).normalized_()]

    # result
    result.success = True
    if frame_ids is None:
        matrix_solutions_x = matrix_solutions_x[0]
        matrix_solutions_y = matrix_solutions_y[0]
        result.calib = TransformPair(x=matrix_solutions_x, y=matrix_solutions_y)
    else:
        matrix_solutions_x = {f: sol for f, sol, in zip(frame_ids.x, matrix_solutions_x)}
        matrix_solutions_y = {f: sol for f, sol, in zip(frame_ids.y, matrix_solutions_y)}
        result.calib = MultiTransformPair(x=matrix_solutions_x, y=matrix_solutions_y)
    return result
