import numpy as np


def gen_Q(points_a, lines_b, normalize=False):
    # check points
    assert points_a.shape[0] == 3
    assert points_a.shape[1] == len(lines_b)

    # iterate points
    Qlist = list()
    for sample_idx in range(points_a.shape[1]):
        # get data
        x = points_a[:, sample_idx]
        y = lines_b[sample_idx].point
        v = lines_b[sample_idx].direction
        assert y.ndim == 1 and len(y) == 3
        assert v.ndim == 1 and len(v) == 3
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
