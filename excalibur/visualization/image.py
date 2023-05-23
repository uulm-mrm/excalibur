import cv2
import motion3d as m3d
import numpy as np


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


def draw_frame_axes(img, poses, length, intrinsics):
    if not isinstance(poses, list):
        poses = [poses]
    for pose in poses:
        pose_mat = pose.asType(m3d.TransformType.kMatrix)
        # check if in front of camera
        if pose_mat.getTranslation()[2] < 4.0:
            continue

        # draw
        cv2.drawFrameAxes(img, intrinsics.camera_matrix, intrinsics.dist_coeffs,
                          pose_mat.getRotationMatrix(), pose_mat.getTranslation(),
                          length=length)
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
