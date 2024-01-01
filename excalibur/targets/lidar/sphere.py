from dataclasses import dataclass
from typing import Optional

import numpy as np

from excalibur.io.dataclass import DataclassIO
from excalibur.fitting.sphere import sphere_ransac

from .utils import EdgeSegmentId, lidar_edge_segmentation


@dataclass
class LidarSphereConfig(DataclassIO):
    """Configuration for lidar sphere detection."""
    # sphere
    sphere_radius: float  #: radius of the detected sphere [m]
    sphere_radius_thresh: float = 0.02  #: radius threshold for a valid sphere [m]
    # edge segmentation
    grad_thresh: float = 1.0  #: minimum depth gradient for detecting foreground plateaus [m]
    min_range: Optional[float] = None  #: cloud points closer than min_range are ignored [m]
    max_range: Optional[float] = None  #: cloud points farther than max_range are ignored [m]
    min_plateau_len: Optional[float] = None  #: minimum plateau length, e.g., for ignoring narrow posts [m]
    # ransac
    ransac_niter: int = 1000  #: number of RANSAC iterations
    ransac_dist_thresh: float = 0.02  #: inlier threshold [m]


def detect_sphere(cloud, cfg: LidarSphereConfig):
    # process input
    min_radius = cfg.sphere_radius - cfg.sphere_radius_thresh
    max_radius = cfg.sphere_radius + cfg.sphere_radius_thresh

    # edge segmentation
    segmentation = lidar_edge_segmentation(
        cloud, grad_thresh=cfg.grad_thresh, min_range=cfg.min_range, max_range=cfg.max_range,
        max_plateau_len=2 * max_radius)

    # sphere detection from plateau and edge points
    is_plateau = segmentation != EdgeSegmentId.NONE
    plateau_xyz = cloud[:, :, :3][is_plateau].copy()

    # sphere detection
    final_sphere = None
    while True:
        # fit sphere
        sphere, sphere_inliers = sphere_ransac(
            plateau_xyz, n_iter=cfg.ransac_niter, dist_thresh=cfg.ransac_dist_thresh,
            sample_range=2 * max_radius, min_radius=min_radius, max_radius=max_radius)
        if sphere is None:
            break

        # check sphere radius
        if min_radius <= sphere.radius <= max_radius:
            final_sphere = sphere
            break

        # remove inliers
        plateau_xyz = np.delete(plateau_xyz, sphere_inliers, axis=0)

    return final_sphere


def detect_sphere_guided(cloud, mask_center, mask_radius, cfg: LidarSphereConfig):
    # process input
    min_radius = cfg.sphere_radius - cfg.sphere_radius_thresh
    max_radius = cfg.sphere_radius + cfg.sphere_radius_thresh

    # mask cloud around selected point
    xyz = cloud[..., :3]
    mask = np.linalg.norm(xyz - mask_center, axis=-1) <= mask_radius
    xyz_masked = xyz[mask]

    # fit sphere
    sphere, sphere_inliers = sphere_ransac(
        xyz_masked, n_iter=cfg.ransac_niter, dist_thresh=cfg.ransac_dist_thresh,
        sample_range=2 * max_radius, min_radius=min_radius, max_radius=max_radius)
    if sphere is None:
        return None

    # get inliers within original cloud
    mask_indices = np.stack(np.where(mask)).T
    sphere.inliers = mask_indices[sphere_inliers, :]

    return sphere
