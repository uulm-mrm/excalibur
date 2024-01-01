from typing import Any

import cv2
import matplotlib
import motion3d as m3d
import numpy as np

from excalibur.utils.image import project_cc_to_ic


def draw_corners(img, corner_list, radius, circle_thickness=-1, line_thickness=1):
    if not isinstance(corner_list, list):
        corner_list = [corner_list]
    for corners in corner_list:
        for row in range(corners.shape[0]):
            if row < corners.shape[0] - 1:
                cv2.line(img, corners[row, :].astype(int), corners[row + 1, :].astype(int),
                         (255, 0, 0), line_thickness)
            color = (0, 255, 0) if row == 0 else (0, 0, 255)
            cv2.circle(img, corners[row, :].astype(int), radius, color, circle_thickness)
    return img


def draw_frame_axes(img, poses, length, intrinsics, min_dist=0.0, thickness=3,
                    color_x=(255, 0, 0), color_y=(0, 255, 0), color_z=(0, 0, 255)):
    # check image
    if not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] in [1, 3, 4])):
        raise RuntimeError("Number of channels must be 1, 3 or 4")
    if img.size == 0:
        raise RuntimeError("Image must not be empty")

    # adjust input
    if not isinstance(poses, list):
        poses = [poses]

    # generate axes points
    axes_points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])

    # iterate poses
    for pose in poses:
        # get pose matrix
        pose_mat = pose.asType(m3d.TransformType.kMatrix)

        # check if in front of camera
        if pose_mat.getTranslation()[2] <= min_dist:
            return img

        # project axes points
        img_points, _ = cv2.projectPoints(axes_points, pose_mat.getRotationMatrix(), pose_mat.getTranslation(),
                                          intrinsics.camera_matrix, intrinsics.dist_coeffs)
        img_points = img_points[:, 0, :].astype(int)

        # draw lines
        cv2.line(img, img_points[0, :], img_points[1, :], color_x, thickness)
        cv2.line(img, img_points[0, :], img_points[2, :], color_y, thickness)
        cv2.line(img, img_points[0, :], img_points[3, :], color_z, thickness)

    return img


def draw_frame_box(img, pose, dimensions, intrinsics, color, thickness=2):
    # check if in front of camera
    pose_mat = pose.asType(m3d.TransformType.kMatrix)
    if pose_mat.getTranslation()[2] < 4.0:
        return img

    # box points in frame coordinates
    x0 = dimensions[0][0]
    x1 = dimensions[0][1]
    y0 = dimensions[1][0]
    y1 = dimensions[1][1]
    z0 = dimensions[2][0]
    z1 = dimensions[2][1]
    box_points_fc = np.array([
        [x0, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        [x1, y0, z0],
        [x0, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
        [x1, y0, z1],
    ])

    # transform box points to image coordinates
    box_points_gc = pose.transformCloud(box_points_fc.T).T
    box_points_ic, _ = cv2.projectPoints(box_points_gc, rvec=np.zeros(3), tvec=np.zeros(3),
                                         cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs)
    box_points_ic = box_points_ic.squeeze().astype(int)

    # draw lines
    box_lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for idx1, idx2 in box_lines:
        cv2.line(img, box_points_ic[idx1], box_points_ic[idx2], color, thickness=thickness)

    return img


def draw_point_cloud(img, intrinsics, cloud: np.ndarray, z_min: float = 0.01, z_max: float = 100.0,
                     radius: int = 1, cmap: Any = 'hsv'):
    # check input
    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError("Image must be in RGB format.")
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise RuntimeError("Cloud must have shape n x 3.")

    # colormap
    cmap = matplotlib.colormaps.get_cmap(cmap)

    # crop points to z range and sort
    cloud = cloud[(cloud[:, 2] >= z_min) & (cloud[:, 2] <= z_max), :]
    cloud = cloud[(-cloud[:, 2]).argsort()]

    # project cloud
    cloud_2d = project_cc_to_ic(cloud, intrinsics)

    # crop points to img range
    cloud_px = cloud_2d.astype(int)
    cloud_inliers = (cloud_px[:, 0] >= 0) & (cloud_px[:, 0] <= img.shape[1]) & \
                    (cloud_px[:, 1] >= 0) & (cloud_px[:, 1] <= img.shape[0])
    cloud = cloud[cloud_inliers]
    cloud_px = cloud_px[cloud_inliers]

    # get colors
    cmap_values = (cloud[:, 2] - z_min) / (z_max - z_min)
    colors = np.array(cmap(cmap_values)) * 255

    # draw cloud on image
    for row in range(cloud_px.shape[0]):
        img = cv2.circle(img, cloud_px[row], radius=radius, color=colors[row], thickness=-1, lineType=cv2.LINE_AA)

    return img
