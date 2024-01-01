from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import motion3d as m3d
import numpy as np

from excalibur.io.dataclass import DataclassIO
from excalibur.utils.math import sorted_eig


@dataclass
class Plane(DataclassIO):
    # 3D plane in Hesse normal form
    normal: np.ndarray
    distance: float
    ex: Optional[np.ndarray] = None
    ey: Optional[np.ndarray] = None
    inliers: Optional[np.ndarray] = None

    @classmethod
    def load(cls, filename: Union[str, Path]):
        return cls.from_json(filename)

    def save(self, filename: Union[str, Path]):
        return self.to_json(filename)

    def get_pose(self):
        # the plane pose, e.g., for visualization in sensor coordinates
        return self.get_transform().inverse()

    def get_transform(self):
        # the plane transformation used, e.g., for planar hand-eye calibration
        target_axis = np.array([0.0, 0.0, 1.0])
        n = np.cross(self.normal, target_axis)
        phi = np.arccos(np.dot(self.normal, target_axis))
        if phi < 1e-6:
            n = np.array([1.0, 0.0, 0.0])
            phi = 0.0
        translation = np.array([0, 0, -self.distance])
        return m3d.AxisAngleTransform(translation, phi, n / np.linalg.norm(n)).normalized_()

    def invert_normal(self):
        self.normal *= -1.0
        self.distance *= -1.0
        self.ey *= -1.0

    def transform(self, t: m3d.TransformInterface):
        plane_point_new = t.transformPoint(self.normal * self.distance)
        plane_normal_new = t.asType(m3d.TransformType.kMatrix).getRotationMatrix() @ self.normal
        plane_distance_new = np.dot(plane_point_new, plane_normal_new)
        return Plane(plane_normal_new, plane_distance_new)

    def calc_distance(self, point: np.ndarray) -> np.ndarray:
        return np.abs(point @ self.normal - self.distance)

    def project(self, point: np.ndarray) -> np.ndarray:
        proj_x = self.ex @ point.T
        if point.shape[-1] == 2:
            return proj_x
        proj_y = self.ey @ point.T
        data = np.column_stack((proj_x, proj_y))
        if point.ndim == 1:
            data = data[0]
        return data

    def project3d(self, point: np.ndarray) -> np.ndarray:
        return self.reproject(self.project(point))

    def reproject(self, point: np.ndarray) -> np.ndarray:
        base_point = self.normal * self.distance
        if point.shape[-1] == 1:
            if point.ndim == 1:
                return self.ex * point[0] + base_point
            else:
                return np.einsum('i,j->ji', self.ex, point[:, 0]) + \
                       base_point
        else:
            if point.ndim == 1:
                return self.ex * point[0] + self.ey * point[1] + base_point
            else:
                return np.einsum('i,j->ji', self.ex, point[:, 0]) + \
                       np.einsum('i,j->ji', self.ey, point[:, 1]) + \
                       base_point


def fit_plane(points: np.ndarray, normal_direction: Optional[Union[List, np.ndarray]] = None) -> Plane:
    # principal component analysis
    points_mean = np.mean(points, axis=0)
    points_adjust = points - points_mean

    # sorted eigenvalues and -vectors of covariance matrix
    cov_mat = np.cov(points_adjust.T)
    _, vecs = sorted_eig(cov_mat)

    # eigenvector with the smallest eigenvalue is plane normal
    normal = vecs[:, 0]
    ex = vecs[:, -1]
    ey = vecs[:, 1] if vecs.shape[1] == 3 else None

    # distance for Hesse normal form
    distance = float(np.dot(normal, points_mean))

    # plane
    plane = Plane(normal=normal, distance=distance, ex=ex, ey=ey)

    # check normal direction
    if normal_direction is not None and plane.normal.dot(normal_direction) < 0.0:
        plane.invert_normal()

    return plane


def plane_ransac(points, n_iter, dist_thresh, sample_range=None,
                 normal_direction: Optional[Union[List, np.ndarray]] = None) \
        -> Tuple[Optional[Plane], Optional[List[int]]]:
    assert points.ndim == 2 and points.shape[1] == 2 or points.shape[1] == 3
    point_dim = points.shape[1]

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
            if len(range_indices) < point_dim:
                # not enough points in range
                continue
            points_subset = points[range_indices]

        # sample points and calculate plane
        if point_dim > points_subset.shape[0]:
            continue
        sample_indices = np.random.choice(points_subset.shape[0], size=point_dim, replace=False)
        plane = fit_plane(points_subset[sample_indices])

        # determine inliers
        distances = plane.calc_distance(points_subset)
        inlier_indices, = np.where(distances < dist_thresh)
        assert len(inlier_indices) >= point_dim

        # add to inliers
        if best_inliers is None or len(inlier_indices) > len(best_inliers):
            if range_indices is not None:
                inlier_indices = range_indices[inlier_indices]
            best_inliers = inlier_indices

    # final estimation with best inliers
    if best_inliers is not None:
        best_plane = fit_plane(points[best_inliers])

        # check normal direction
        if normal_direction is not None and best_plane.normal.dot(normal_direction) < 0.0:
            best_plane.invert_normal()
    else:
        best_plane = None

    return best_plane, best_inliers
