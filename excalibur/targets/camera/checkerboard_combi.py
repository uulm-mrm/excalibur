from dataclasses import dataclass, field
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np

from excalibur.io.dataclass import DataclassIO
from excalibur.utils.image import project_cc_to_ic

from .aruco import ArucoConfig, detect_aruco
from .checkerboard import CheckerboardConfig, detect_checkerboard_corners, estimate_checkerboard_pose_multi
from .utils import fix_aruco_params, MarkerDetection


@dataclass
class ArucoMarker(DataclassIO):
    """Single ArUco marker configuration for combined ArUco-checkerboard detection."""
    length: float  #: marker side length [m]
    pose: m3d.TransformInterface  #: marker pose w.r.t. the checkerboard


@dataclass
class CheckerboardCombiConfig(DataclassIO):
    """Configuration for combined checkerboard-ArUco detection."""
    name: str  #: unique checkerboard identifier string
    checkerboard_cfg: CheckerboardConfig  #: checkerboard detection configuration
    dict_id: int  #: aruco dictionary identifier (e.g., 0 for cv2.aruco.DICT_4X4_50)
    markers: Dict[int, ArucoMarker]  #: available aruco markers with identifier as key
    cutout_margin: float  #: margin for the checkerboard cutout w.r.t. the relative aruco pose [m]
    force_z_up: False  #: force the checkerboard z-axis to point upwards in positive z-direction
    aruco_params: cv2.aruco.DetectorParameters = field(default_factory=cv2.aruco.DetectorParameters)  #: parameters for aruco detection

    @classmethod
    def from_dict(cls, data):
        config = super().from_dict(data)
        config.aruco_params = fix_aruco_params(config.aruco_params)
        return config


def _dq_error_squared(x):
    v = x.asType(m3d.TransformType.kDualQuaternion).toArray()
    if v[0] < 0:
        v *= -1
    err = v - m3d.DualQuaternionTransform().toArray()
    return err @ err


def detect_checherboard_combi(img: np.ndarray, cfg: CheckerboardCombiConfig, intrinsics, debug=False):
    # aruco detection
    aruco_cfg = ArucoConfig(
        dict_id=cfg.dict_id,
        marker_length={aruco_id: marker.length for aruco_id, marker in cfg.markers.items()},
        params=cfg.aruco_params
    )
    aruco_detections = detect_aruco(img, aruco_cfg, intrinsics=intrinsics)
    if len(aruco_detections) == 0:
        return None, aruco_detections

    # find image dimensions
    cb_cfg = cfg.checkerboard_cfg
    cb_size = (cb_cfg.board_dim[0] * cb_cfg.square_length, cb_cfg.board_dim[1] * cb_cfg.square_length)
    box_x2 = cb_size[0] / 2.0 + cfg.cutout_margin
    box_y2 = cb_size[1] / 2.0 + cfg.cutout_margin
    box_z2 = 0.1 + cfg.cutout_margin
    box_points = np.array([
        [box_x2, box_y2, box_z2],
        [-box_x2, box_y2, box_z2],
        [box_x2, -box_y2, box_z2],
        [-box_x2, -box_y2, box_z2],
        [box_x2, box_y2, -box_z2],
        [-box_x2, box_y2, -box_z2],
        [box_x2, -box_y2, -box_z2],
        [-box_x2, -box_y2, -box_z2],
    ])

    # box points in image coordinates
    box_points_ic_list = []
    for detection in aruco_detections:
        if detection.identifier not in cfg.markers:
            continue
        for pose in detection.pose_options:
            # transform and project
            origin_pose = pose * cfg.markers[detection.identifier].pose.inverse()
            box_points_cc = origin_pose.transformCloud(box_points.T).T
            box_points_ic, _ = cv2.projectPoints(box_points_cc, rvec=np.zeros(3), tvec=np.zeros(3),
                                                 cameraMatrix=intrinsics.camera_matrix,
                                                 distCoeffs=intrinsics.dist_coeffs)
            box_points_ic = box_points_ic.squeeze()

            # handle out of image points
            box_points_ic[box_points_ic < 0] = 0
            box_points_ic[box_points_ic[:, 0] > img.shape[1], 0] = img.shape[1]
            box_points_ic[box_points_ic[:, 1] > img.shape[0], 1] = img.shape[0]

            # append
            box_points_ic_list.append(box_points_ic)

    if len(box_points_ic_list) == 0:
        return None, aruco_detections

    # range
    cb_corners = None
    for box_points_idx, box_points_ic in enumerate(box_points_ic_list):
        ic_min = np.floor(np.min(box_points_ic, axis=0)).astype(int)
        ic_max = np.ceil(np.max(box_points_ic, axis=0)).astype(int)

        # cut out
        img_cut = img[ic_min[1]:ic_max[1], ic_min[0]:ic_max[0]]
        if img_cut.shape[0] < 3 or img_cut.shape[1] < 3:
            continue

        if debug:
            plt.subplot(2, len(box_points_ic_list), box_points_idx + 1)
            plt.title(f"Cutout {box_points_idx + 1}/{len(box_points_ic_list)}")
            plt.imshow(img_cut, cmap='gray')

        # checkerboard corners
        cb_corners = detect_checkerboard_corners(img_cut, cfg.checkerboard_cfg.board_dim)
        if cb_corners is not None:
            cb_corners += ic_min
            break  # one checkerboard detection is enough

    # exit if no corners were found
    if cb_corners is None:
        plt.show()
        return None, aruco_detections

    if debug:
        plt.subplot(2, 1, 2)
        img_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(img_corners, cfg.checkerboard_cfg.board_dim, cb_corners, True)
        plt.imshow(img_corners)
        plt.title("Checkerboard Corners")
        plt.show()

    # checkerboard poses
    cb_pose_cc_a, cb_pose_cc_b = estimate_checkerboard_pose_multi(
        cb_corners, cfg.checkerboard_cfg.board_dim, cfg.checkerboard_cfg.square_length, intrinsics)
    rot_z = m3d.EulerTransform([0, 0, 0], 0, 0, np.pi, m3d.EulerAxes.kSXYZ)
    cb_poses_cc = [cb_pose_cc_a, cb_pose_cc_b, cb_pose_cc_a * rot_z, cb_pose_cc_b * rot_z]

    # force z up
    if cfg.force_z_up:
        cb_poses_cc_up = []
        for pose in cb_poses_cc:
            # get z-axis direction in image coordinates
            z_axis_points = np.array([[0, 0, 0], [0, 0, 1]])
            z_axis_points_cc = pose.transformCloud(z_axis_points.T).T
            z_axis_points_ic = project_cc_to_ic(z_axis_points_cc, intrinsics)
            z_axis_ic = z_axis_points_ic[1, :] - z_axis_points_ic[0, :]

            # check if it is pointing upwards
            if z_axis_ic[1] < 0:
                cb_poses_cc_up.append(pose)

        cb_poses_cc = cb_poses_cc_up

    # check poses
    if len(cb_poses_cc) == 0:
        return None, aruco_detections

    # select best pose candidate
    consistency_errors = []
    for detection in aruco_detections:
        if detection.identifier not in cfg.markers:
            continue
        consistency_errors.append(
            [[_dq_error_squared(cp * cfg.markers[detection.identifier].pose * pose.inverse())
              for pose in detection.pose_options]
             for cp in cb_poses_cc]
        )
    consistency_errors = np.stack(consistency_errors)

    # select best checkerboard
    cb_consistency_errors = np.sum(np.min(consistency_errors, axis=2), axis=0)
    best_cb_index = np.argmin(cb_consistency_errors)
    best_cb_pose_cc = cb_poses_cc[best_cb_index]

    # select best aruco poses
    best_aruco_detections = []
    aruco_consistency_errors = consistency_errors[:, best_cb_index, :]
    aruco_index = 0
    for detection in aruco_detections:
        if detection.identifier not in cfg.markers:
            continue
        best_aruco_pose_index = np.argmin(aruco_consistency_errors[aruco_index])
        detection.pose = detection.pose_options[best_aruco_pose_index]
        aruco_index += 1
        best_aruco_detections.append(detection)

    # result
    checkerboard_detection = MarkerDetection(corners=cb_corners, pose=best_cb_pose_cc)
    return checkerboard_detection, best_aruco_detections
