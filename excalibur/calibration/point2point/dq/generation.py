import motion3d as m3d
import numpy as np


def gen_Mlist(points_a, points_b):
    # check points
    assert points_a.shape[0] == points_b.shape[0] == 3
    assert points_a.shape[1] == points_b.shape[1]

    # iterate points
    Mlist = list()
    for point_idx in range(points_a.shape[1]):
        # dual part matrices
        Da = m3d.Quaternion(0, *points_a[:, point_idx]).toPositiveMatrix()
        Db = m3d.Quaternion(0, *points_b[:, point_idx]).toNegativeMatrix()

        # store result
        Mlist.append(np.column_stack([Da - Db, - 2 * np.eye(4)]))

    return Mlist
