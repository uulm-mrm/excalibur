import numpy as np


def gen_Q(points_a, line_vecs_b, line_origins_b=None, normalize=False):
    # check points
    assert points_a.shape[0] == line_vecs_b.shape[0] == 3
    assert points_a.shape[1] == line_vecs_b.shape[1]
    assert line_origins_b is None or (line_origins_b.shape[0] == 3 and line_origins_b.shape[1] == line_vecs_b.shape[1])

    # iterate points
    Qlist = list()
    for point_idx in range(points_a.shape[1]):
        # get data
        x = points_a[:, point_idx]
        if line_origins_b is None:
            y = np.zeros(3)
        else:
            y = line_origins_b[:, point_idx]
        v = line_vecs_b[:, point_idx]
        assert np.abs(np.dot(v, v) - 1.0) < 1e-6

        # distance matrix
        C = np.eye(3) - np.outer(v, v)

        # cost matrix
        xs = np.concatenate([x, [1.0]])
        N = np.hstack([np.kron(xs.reshape(1, 4), np.eye(3)), -y.reshape(3, 1)])
        M = N.T @ C @ N

        # store matrix
        Qlist.append(M)

    # output
    if normalize:
        return np.mean(Qlist, axis=0)
    else:
        return np.sum(Qlist, axis=0)
