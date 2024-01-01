from dataclasses import dataclass, field
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np

from excalibur.io.dataclass import DataclassIO

from .utils import fix_aruco_params, MarkerDetection


@dataclass
class CharucoBoardConfig(DataclassIO):
    """Configuration for ChArUco board detection."""
    dict_id: int  #: aruco dictionary identifier (e.g., 0 for cv2.aruco.DICT_4X4_50)
    board_dim: Tuple[int, int]  #: number of cells (length/x, width/y)
    square_length: float  #: cell size [m]
    marker_length: float  #: marker side length [m]
    aruco_params: cv2.aruco.DetectorParameters = field(default_factory=cv2.aruco.DetectorParameters)  # aruco detection parameters

    @classmethod
    def from_dict(cls, data):
        config = super().from_dict(data)
        config.aruco_params = fix_aruco_params(config.aruco_params)
        return config


def detect_charuco(img: np.ndarray, cfg: CharucoBoardConfig, intrinsics=None, aruco_blacklist=None, debug=False):
    # detect aruco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cfg.dict_id)
    aruco_corners, aruco_ids, _rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=cfg.aruco_params)

    if debug:
        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.aruco.drawDetectedMarkers(img_copy, aruco_corners, aruco_ids)
        plt.figure()
        plt.subplot(211)
        plt.imshow(img_copy)

    if aruco_ids is None:
        if debug:
            plt.show()
        return None

    if aruco_blacklist is not None:
        is_whitelist = [aid not in aruco_blacklist for aid in aruco_ids]
        if np.any(is_whitelist):
            return None
        aruco_corners = [ac for ac, is_wl in zip(aruco_corners, is_whitelist) if is_wl]
        aruco_ids = aruco_ids[is_whitelist]

    # detect board
    # TODO(horn): add RANSAC for handling duplicate IDs and outliers
    charuco_board = cv2.aruco.CharucoBoard(cfg.board_dim, cfg.square_length, cfg.marker_length, aruco_dict)
    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        aruco_corners, aruco_ids, img, charuco_board,
        cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs)

    if debug:
        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids)
        plt.subplot(212)
        plt.imshow(img_copy)
        plt.show()

    # at least 6 corners are required for estimatePoseCharucoBoard
    if num_corners < 6:
        return None

    # estimate pose
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, charuco_board,
        cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs, rvec=None, tvec=None)
    if not success or rvec is None or tvec is None:
        return None
    pose = m3d.MatrixTransform(tvec.flatten(), cv2.Rodrigues(rvec.flatten())[0])

    # adjust translation
    center_translation = m3d.EulerTransform(
        [cfg.board_dim[0] * cfg.square_length / 2, cfg.board_dim[1] * cfg.square_length / 2, 0.0], [0.0, 0.0, 0.0])
    pose *= center_translation

    # adjust rotation axes (z pointing towards camera)
    rot = pose.getRotationMatrix()
    new_rot = np.column_stack((rot[:, 0], -rot[:, 1], -rot[:, 2]))
    pose.setRotationMatrix(new_rot)

    return MarkerDetection(
        corners=charuco_corners,
        identifier=np.min(charuco_ids),
        pose=pose,
    )
