import time

import motion3d as m3d
import numpy as np

from ...base import CalibrationResult


def solve_horn_quat(points_a, points_b):
    assert points_a.shape[0] == points_b.shape[0] == 3 and points_a.shape[1] == points_b.shape[1]

    # initialize result
    result = CalibrationResult()
    start_time = time.time()

    # calculate and subtract centroids
    centroid_a = np.mean(points_a, axis=1, keepdims=True)
    centroid_b = np.mean(points_b, axis=1, keepdims=True)
    points_a_sub = points_a - centroid_a
    points_b_sub = points_b - centroid_b

    # calculate M matrix (a and b are interchanged here compared Horn)
    M = np.dot(points_b_sub, points_a_sub.T)

    # create N matrix
    Sxx, Sxy, Sxz = M[0, 0], M[0, 1], M[0, 2]
    Syx, Syy, Syz = M[1, 0], M[1, 1], M[1, 2]
    Szx, Szy, Szz = M[2, 0], M[2, 1], M[2, 2]

    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
        [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
        [Szx - Sxz, Sxy + Syx, Syy - Sxx - Szz, Syz + Szy],
        [Sxy - Syx, Szx + Sxz, Syz + Szy, Szz - Sxx - Syy]
    ])

    # eigendecomposition of N
    vals, vecs = np.linalg.eig(N)
    vals = np.real(vals)
    vecs = np.real(vecs)

    # largest positive eigenvalue
    max_idx = np.argmax(vals)
    max_vec = vecs[:, max_idx]
    quat = m3d.Quaternion.FromArray(max_vec)  # solution quaternion is eigenvector corresponding to maximum eigenvalue

    # calculate translation
    t = centroid_a.squeeze() - quat.transformPoint(centroid_b)
    result.run_time = time.time() - start_time

    # create transformation
    solution = m3d.QuaternionTransform(t, quat, unsafe=True).normalized_()

    # create result
    result = CalibrationResult()
    result.success = True
    result.calib = solution
    return result
