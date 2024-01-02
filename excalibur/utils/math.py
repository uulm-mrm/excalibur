import numpy as np


def canonical_vector(dim, u):
    x = np.zeros(dim)
    x[u] = 1.0
    return x


def canonical_matrix(dim, u, v):
    x = np.zeros((dim, dim))
    x[u, v] = 1.0
    return x


def cross_product_matrix(x):
    assert len(x) == 3
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0],
    ])


def rmse(x):
    return np.sqrt(np.mean(np.array(x) ** 2))


def schur_complement_A(M, rows, cols):
    A = M[:rows, :cols]
    B = M[:rows, cols:]
    C = M[rows:, :cols]
    D = M[rows:, cols:]
    return D - C @ np.linalg.inv(A) @ B


def schur_complement_D(M, rows, cols):
    A = M[:rows, :cols]
    B = M[:rows, cols:]
    C = M[rows:, :cols]
    D = M[rows:, cols:]
    return A - B @ np.linalg.inv(D) @ C


def submat(Q, rows, cols):
    return Q[rows, :][:, cols]


def schur_complement_indices(M, main_rows, main_cols, comp_rows, comp_cols):
    return submat(M, main_rows, main_cols) - submat(M, main_rows, comp_cols) @ \
           np.linalg.inv(submat(M, comp_rows, comp_cols)) @ submat(M, comp_rows, main_cols)


def sorted_eig(a):
    vals, vecs = np.linalg.eig(a)
    sort_indices = np.argsort(np.abs(vals))
    vals = np.abs(vals[sort_indices])
    vecs = np.real(vecs[:, sort_indices])
    return vals, vecs
