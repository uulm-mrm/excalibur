import numpy as np

from excalibur.utils.logging import logger


def calculate_nullspace_factors_daniilidis(nullspace, real_indices, dual_indices):
    assert nullspace.shape[1] == 2

    # split nullspace
    nullspace_real = nullspace[real_indices, :]
    nullspace_dual = nullspace[dual_indices, :]

    # separate real and dual part
    u1 = nullspace_real[:, 0]
    v1 = nullspace_dual[:, 0]
    u2 = nullspace_real[:, 1]
    v2 = nullspace_dual[:, 1]
    order_changed = False

    # change order if necessary
    if abs(u1.T @ v1) < abs(u2.T @ v2):
        u1, u2 = u2, u1
        v1, v2 = v2, v1
        order_changed = True

    # a, b, c of eq 34
    a1 = u1.T @ u1
    b1 = 2 * u1.T @ u2
    c1 = u2.T @ u2

    # a, b, c of eq 35
    a2 = u1.T @ v1
    b2 = u1.T @ v2 + u2.T @ v1
    c2 = u2.T @ v2

    # solve eq 35
    disc2_square = b2 ** 2 - 4 * a2 * c2
    disc2 = np.sqrt(np.abs(disc2_square))
    s1 = (-b2 + disc2) / (2 * a2)
    s2 = (-b2 - disc2) / (2 * a2)

    # select s corresponding two largest value of eq 34
    value1 = s1 ** 2 * a1 + s1 * b1 + c1
    value2 = s2 ** 2 * a1 + s2 * b1 + c1

    if value1 > value2:
        s = s1
        value = value1
    else:
        s = s2
        value = value2

    # check value
    if value < 1e-6:
        logger.info("Nullspace factor calculation (Daniilidis) failed")
        return None

    # calculate lambdas from s
    l2 = np.sqrt(1 / value)
    l1 = s * l2

    # lambdas
    if order_changed:
        return np.array([l2, l1])
    else:
        return np.array([l1, l2])
