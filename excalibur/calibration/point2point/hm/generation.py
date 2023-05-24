import numpy as np


def gen_Q(points_a, points_b, normalize=False):
    # check points
    assert points_a.shape[0] == points_b.shape[0] == 3
    assert points_a.shape[1] == points_b.shape[1]

    # iterate points
    Qlist = list()
    for point_idx in range(points_a.shape[1]):
        # get data
        x = points_a[:, point_idx]
        y = points_b[:, point_idx]

        # cost matrix
        xs = np.concatenate([x, [1.0]])
        N = np.hstack([np.kron(xs.reshape(1, 4), np.eye(3)), -y.reshape(3, 1)])
        M = N.T @ N

        # store matrix
        Qlist.append(M)

    # output
    if normalize:
        return np.mean(Qlist, axis=0)
    else:
        return np.sum(Qlist, axis=0)
