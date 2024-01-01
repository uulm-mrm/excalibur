import numpy as np


def normalize_rotation_matrix(R):
    # determine epsilon
    eps = np.finfo(R.dtype).eps

    # unit determinant
    det = np.linalg.det(R)
    if np.abs(det) < eps:
        return None
    R *= np.cbrt(np.copysign(1.0, det) / np.abs(det))

    # make orthogonal
    u, _, vt = np.linalg.svd(R)
    R = u @ vt

    # fix inversion
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt

    return R
