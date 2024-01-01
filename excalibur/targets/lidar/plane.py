from dataclasses import dataclass
from typing import Optional

import numpy as np

from excalibur.io.dataclass import DataclassIO
from excalibur.fitting.plane import plane_ransac
from excalibur.visualization.cloud import visualize_cloud


@dataclass
class LidarPlaneConfig(DataclassIO):
    """Configuration for lidar plane detection."""
    plane_niter: int = 1000  #: number of RANSAC iterations
    plane_dist_thresh: float = 0.1  #: inlier threshold [m]
    min_dist: float = 1.0  #: cloud points closer than min_dist are ignored [m]
    max_dist: Optional[float] = None  #: cloud points farther than max_dist are ignored [m]


def detect_plane(cloud: np.ndarray, cfg: LidarPlaneConfig, debug=False):
    # prepare cloud
    cloud = cloud[..., :3]
    if cloud.ndim == 3:
        cloud = cloud.reshape((cloud.shape[0] * cloud.shape[1], cloud.shape[2]))

    range = np.linalg.norm(cloud, axis=-1)
    if cfg.max_dist is None:
        range_indices = range >= cfg.min_dist
    else:
        range_indices = (range >= cfg.min_dist) & (range <= cfg.max_dist)
    cloud = cloud[range_indices, :]

    # ransac
    plane, plane_inliers = plane_ransac(cloud[..., :3], n_iter=cfg.plane_niter, dist_thresh=cfg.plane_dist_thresh)

    # show inliers
    if debug:
        colors = np.zeros((cloud.shape[0], 3))
        colors[plane_inliers, :] = [0, 255, 0]
        visualize_cloud(cloud, colors=colors, pose=plane.get_transform())

    # result
    return plane
