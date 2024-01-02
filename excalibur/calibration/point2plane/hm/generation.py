import numpy as np


def gen_Q(points_a, planes_b, normalize=False):
    # check points
    assert points_a.shape[0] == 3
    assert points_a.shape[1] == len(planes_b)

    # iterate points
    Qlist = list()
    for sample_idx in range(points_a.shape[1]):
        # get data
        x = points_a[:, sample_idx]
        y = planes_b[sample_idx].normal * planes_b[sample_idx].distance
        n = planes_b[sample_idx].normal
        assert n.ndim == 1 and len(n) == 3
        assert np.abs(np.dot(n, n) - 1.0) < 1e-6

        # distance matrix
        C = np.outer(n, n)

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
