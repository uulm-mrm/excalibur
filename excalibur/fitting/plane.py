import numpy as np

from excalibur.io.geometry import Plane
from excalibur.utils.math import sorted_eig


def fit_plane(points) -> Plane:
    # principal component analysis
    points_mean = np.mean(points, axis=0)
    points_adjust = points - points_mean

    # sorted eigenvalues and -vectors of covariance matrix
    cov_mat = np.cov(points_adjust.T)
    _, vecs = sorted_eig(cov_mat)

    # eigenvector with the smallest eigenvalue is plane normal
    normal = vecs[:, 0]

    # distance for Hesse normal form
    distance = np.dot(normal, points_mean)

    return Plane(normal=normal, distance=distance)
