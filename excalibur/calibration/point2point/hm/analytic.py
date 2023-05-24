import time

import motion3d as m3d
import numpy as np

from ...base import CalibrationResult


def solve_arun(points_a, points_b):
    assert points_a.shape[0] == points_b.shape[0] == 3 and points_a.shape[1] == points_b.shape[1]

    # initialize result
    result = CalibrationResult()
    start_time = time.time()

    # calculate and subtract centroids
    centroid_a = np.mean(points_a, axis=1, keepdims=True)
    centroid_b = np.mean(points_b, axis=1, keepdims=True)
    points_a_sub = points_a - centroid_a
    points_b_sub = points_b - centroid_b

    # calculate H matrix (a and b are interchanged here compared to Arun)
    H = np.dot(points_b_sub, points_a_sub.T)

    # find the SVD of H
    U, _, Vt = np.linalg.svd(H)
    Ut = U.T
    V = Vt.T

    # calculate X
    X = V @ Ut

    # enforce positive determinant for R
    det_X = np.linalg.det(X)
    is_reflection = det_X < 0.0
    det_fix = np.eye(points_a.shape[0])
    det_fix[-1, -1] = det_X
    R = V @ det_fix @ Ut

    # calculate translation vector and store time
    t = centroid_a - np.dot(R, centroid_b)
    result.run_time = time.time() - start_time

    # create transformation
    solution = m3d.MatrixTransform(t, R, unsafe=True).normalized_()

    # create result
    result = CalibrationResult()
    result.success = True
    result.calib = solution
    result.aux_data = {
        'is_reflection': is_reflection
    }
    return result
