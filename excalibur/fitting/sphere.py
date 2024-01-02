from dataclasses import dataclass
from typing import Optional

import numpy as np

from excalibur.io.dataclass import DataclassIO


@dataclass
class Sphere(DataclassIO):
    center: np.ndarray
    radius: float
    inliers: Optional[np.ndarray] = None

    def calc_distance(self, point: np.ndarray) -> np.ndarray:
        return np.abs(np.linalg.norm(point - self.center, axis=1) - self.radius)


def sorted_eig(a):
    vals, vecs = np.linalg.eig(a)
    sort_indices = np.argsort(np.abs(vals))
    vals = np.abs(vals[sort_indices])
    vecs = np.real(vecs[:, sort_indices])
    return vals, vecs


def fit_sphere(points):
    # https://ieeexplore.ieee.org/document/6312298
    # create equation system
    points_sum = np.sum(points, axis=0, keepdims=True)
    D = np.vstack((
        np.hstack((2 * points_sum, [[points.shape[0]]])),
        np.hstack((2 * points.T @ points, points_sum.T))
    ))
    E = np.concatenate((
        [np.sum(points ** 2)],
        np.sum(points.T @ (points ** 2), axis=1)
    ))

    # solve
    try:
        Q = np.linalg.solve(D, E)
    except np.linalg.LinAlgError:
        return None

    # recover
    center = Q[:-1]
    disc = np.sum(Q[:-1] ** 2) + Q[-1]
    if disc < 0.0:
        return None
    radius = np.sqrt(disc)
    return Sphere(center=center, radius=radius)


def sphere_ransac(points, n_iter, dist_thresh, sample_range=None, min_radius=None, max_radius=None):
    # min points
    min_sphere_points = points.shape[1] + 1
    if points.shape[0] < min_sphere_points:
        return None, None

    # ransac iterations
    best_inliers = None
    for _ in range(n_iter):
        # check sample range
        if sample_range is None:
            range_indices = None
            points_subset = points
        else:
            # sample single point and reduce points to sample range
            first_sample = np.random.randint(low=0, high=points.shape[0])
            range_indices, = np.where(np.linalg.norm(points - points[first_sample], axis=1) <= sample_range)
            if len(range_indices) < min_sphere_points:
                # not enough points in range
                continue
            points_subset = points[range_indices]

        # sample points and calculate sphere
        sample_indices = np.random.choice(points_subset.shape[0], size=min_sphere_points, replace=False)
        sphere = fit_sphere(points_subset[sample_indices])
        if sphere is None:
            continue

        # check radius
        if min_radius is not None and sphere.radius < min_radius:
            continue
        if max_radius is not None and sphere.radius > max_radius:
            continue

        # determine inliers
        distances = sphere.calc_distance(points_subset)
        inlier_indices, = np.where(distances < dist_thresh)
        if len(inlier_indices) < min_sphere_points:
            continue

        # add to inliers
        if best_inliers is None or len(inlier_indices) > len(best_inliers):
            if range_indices is not None:
                inlier_indices = range_indices[inlier_indices]
            best_inliers = inlier_indices

    # final estimation with best inliers
    if best_inliers is not None:
        best_sphere = fit_sphere(points[best_inliers])
    else:
        best_sphere = None
    return best_sphere, best_inliers
