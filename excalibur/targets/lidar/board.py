from dataclasses import dataclass
import itertools
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np

from excalibur.calibration.point2point.hm.analytic import solve_arun
from excalibur.io.dataclass import DataclassIO
from excalibur.fitting.plane import plane_ransac
from excalibur.fitting.sphere import sphere_ransac
from excalibur.visualization.cloud import visualize_cloud

from .utils import EdgeSegmentId, lidar_edge_segmentation


SEGMENTATION_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'segmentation', [(0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
OVERLAY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'overlay', [(1, 0, 0), (0, 1, 0)])


def _check_geometric_consistency(points, ref_points, geom_consist_thresh):
    # initialize output
    best_points = None
    best_errors = None

    # check input
    num_pts = points.shape[0]
    num_ref = ref_points.shape[0]
    if num_pts < num_ref:
        return None, None

    # generate reference lengths
    ref_lengths = [np.linalg.norm(ref_points[idx1] - ref_points[idx2])
                   for idx1, idx2 in itertools.combinations(range(num_ref), 2)]

    # iterate permutations
    for indices in itertools.permutations(range(num_pts), num_ref):
        # generate sample lengths
        sample_points = points[indices, :]
        sample_lengths = [np.linalg.norm(sample_points[idx1] - sample_points[idx2])
                          for idx1, idx2 in itertools.combinations(range(num_ref), 2)]

        # compare lengths
        errors = np.array([np.abs(l1 - l2) for l1, l2 in zip(ref_lengths, sample_lengths)])
        if np.any(errors > geom_consist_thresh):
            continue
        if best_errors is None or np.sum(errors) < np.sum(best_errors):
            best_points = sample_points
            best_errors = errors

    return best_points, best_errors


@dataclass
class LidarBoardConfig(DataclassIO):
    """Configuration for lidar board detection."""
    # board
    ref_centers: np.ndarray  #: (n,2)-array with hole center positions in 2D board coordinates [m]
    circle_radius: float  #: hole/circle radius of all holes [m]
    max_diagonal: float  #: diagonal of the calibration baord [m]
    circle_radius_thresh: float = 0.01  #: radius threshold for a valid hole/circle [m]
    # edge segmentation
    range_interpolation_threshold: float = 0.0  #: missing points with neighbors in between this threshold are interpolated
    grad_thresh: float = 1.0  #: minimum depth gradient for detecting foreground plateaus [m]
    min_range: Optional[float] = None  #: cloud points closer than min_range are ignored [m]
    max_range: Optional[float] = None  #: cloud points farther than max_range are ignored [m]
    min_plateau_len: Optional[float] = None  #: minimum plateau length, e.g., for ignoring narrow posts [m]
    # plane ransac
    plane_niter: int = 100  #: number of RANSAC iterations for plane fitting
    plane_dist_thresh: float = 0.1  #: inlier threshold [m]
    # line ransac
    line_niter: int = 100  #: number of RANSAC iterations for line fitting (used to remove edge points)
    line_dist_thresh: float = 0.01  #: inlier threshold [m]
    min_line_inliers: int = 15  #: minimum number of inliers for removing an edge line
    # circle ransac
    enforce_circle_radius: bool = False  #: enforce circle radius during fitting instead of checking it afterward [m]
    circle_niter: int = 100  #: number of RANSAC iterations for the circle fitting (hole detection)
    circle_dist_thresh: float = 0.02  #: inlier threshold [m]
    # geometric consistency
    geom_consist_thresh: float = 0.06  #: geometric consistency for a valid group of hole/circle detections [m]


def detect_board(cloud: np.ndarray, cfg: LidarBoardConfig, debug=False, debug_3d=False):
    # edge segmentation
    segmentation = lidar_edge_segmentation(
        cloud, grad_thresh=cfg.grad_thresh, min_range=cfg.min_range, max_range=cfg.max_range,
        range_interpolation_threshold=cfg.range_interpolation_threshold,
        min_plateau_len=cfg.min_plateau_len, max_plateau_len=cfg.max_diagonal)

    if debug:
        depth_img = np.linalg.norm(cloud, axis=2)
        plt.figure()
        plt.subplot(411)
        plt.title("Depth Image")
        plt.imshow(depth_img, vmin=cfg.min_range, vmax=cfg.max_range, cmap='hsv')
        plt.subplot(412)
        plt.title("Plateau Segmentation")
        plt.imshow(segmentation, cmap=SEGMENTATION_CMAP)

    if debug_3d:
        colors = np.zeros((*cloud.shape[:-1], 3))
        colors[segmentation == EdgeSegmentId.LEFT_EDGE] = [255, 0, 0]
        colors[segmentation == EdgeSegmentId.PLATEAU] = [0, 255, 0]
        colors[segmentation == EdgeSegmentId.RIGHT_EDGE] = [0, 0, 255]
        visualize_cloud(cloud, colors)

    # plane detection from plateau and edge points
    is_plateau = segmentation != EdgeSegmentId.NONE
    plateau_xyz = cloud[:, :, :3][is_plateau]
    plateau_seg = segmentation[is_plateau]
    if len(plateau_xyz) == 0:
        if debug:
            plt.show()
        return None
    plane, plane_inliers = plane_ransac(
        plateau_xyz, n_iter=cfg.plane_niter, dist_thresh=cfg.plane_dist_thresh, sample_range=cfg.max_diagonal)

    if debug:
        # overlay with highlighted plateau inliers
        plateau_overlay = np.zeros(plateau_xyz.shape[0])
        plateau_overlay[plane_inliers] = 1.0
        img_overlay = np.zeros(np.prod(depth_img.shape))
        img_overlay[is_plateau.flatten()] = plateau_overlay
        img_overlay = img_overlay.reshape(depth_img.shape)

        plt.subplot(413)
        plt.title("Plane Inliers")
        plt.imshow(depth_img, vmin=cfg.min_range, vmax=cfg.max_range, cmap='gray')
        plt.imshow(img_overlay, cmap=OVERLAY_CMAP, alpha=0.5)

    if debug_3d:
        colors = np.zeros((plateau_xyz.shape[0], 3))
        colors[plane_inliers, :] = [255.0, 0, 0]
        visualize_cloud(plateau_xyz, colors)

    # reduce to inliers
    plane_xyz = plateau_xyz[plane_inliers]
    plane_seg = plateau_seg[plane_inliers]

    # reduce to edges and project onto plane
    plane_edges_xyz = plane_xyz[(plane_seg == EdgeSegmentId.LEFT_EDGE) | (plane_seg == EdgeSegmentId.RIGHT_EDGE)]
    plane_edges_2d_orig = plane.project(plane_edges_xyz)
    plane_edges_2d = plane_edges_2d_orig.copy()

    # remove lines
    while True:
        # fit line
        _, line_inliers = plane_ransac(
            plane_edges_2d, n_iter=cfg.line_niter, dist_thresh=cfg.line_dist_thresh)
        if line_inliers is None or len(line_inliers) < cfg.min_line_inliers:
            break

        # remove inliers
        plane_edges_2d = np.delete(plane_edges_2d, line_inliers, axis=0)
    plane_edges_2d_no_lines = plane_edges_2d.copy()

    # circle detection
    min_radius = cfg.circle_radius - cfg.circle_radius_thresh
    max_radius = cfg.circle_radius + cfg.circle_radius_thresh
    if cfg.enforce_circle_radius:
        sphere_ransac_kwargs = {
            'min_radius': min_radius,
            'max_radius': max_radius,
        }
    else:
        sphere_ransac_kwargs = {}

    circles = []
    while True:
        # fit circle
        circle, circle_inliers = sphere_ransac(
            plane_edges_2d, n_iter=cfg.circle_niter, dist_thresh=cfg.circle_dist_thresh, **sphere_ransac_kwargs)
        if circle is None:
            break
        circles.append(circle)

        # remove inliers
        plane_edges_2d = np.delete(plane_edges_2d, circle_inliers, axis=0)

    # check circle radius
    centers = np.array([c.center for c in circles if min_radius <= c.radius <= max_radius])

    if debug:
        plt.subplot(414)
        plt.title("Plane Edges and Circle Fitting")
        plt.plot(plane_edges_2d_orig[:, 0], plane_edges_2d_orig[:, 1], 'k.', label='edges (lines)')
        plt.plot(plane_edges_2d_no_lines[:, 0], plane_edges_2d_no_lines[:, 1], 'b.', label='edges (no lines)')
        if len(circles) > 0:
            all_centers = np.array([c.center for c in circles])
            plt.plot(all_centers[:, 0], all_centers[:, 1], 'ro', label='centers (invalid radius)')
        if len(centers) > 0:
            plt.plot(centers[:, 0], centers[:, 1], 'go', label='centers (valid radius)')
        plt.axis('equal')
        plt.legend()
        plt.show()

    # geometric consistency
    best_centers, best_errors = _check_geometric_consistency(
        centers, cfg.ref_centers, geom_consist_thresh=cfg.geom_consist_thresh)

    # reproject circle centers to 3D
    if best_centers is None:
        return None
    best_centers_3d = plane.reproject(best_centers)

    # estimate board pose
    ref_centers_3d = np.zeros((cfg.ref_centers.shape[0], 3))
    ref_centers_3d[:, :2] = cfg.ref_centers
    result = solve_arun(best_centers_3d.T, ref_centers_3d.T)
    board_pose = result.calib.asType(m3d.TransformType.kMatrix)

    return board_pose
