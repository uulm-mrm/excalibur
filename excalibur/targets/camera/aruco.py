from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import cv2
import motion3d as m3d
import numpy as np

from excalibur.io.dataclass import DataclassIO

from .utils import fix_aruco_params, MarkerDetection


@dataclass
class ArucoConfig(DataclassIO):
    """Configuration for ArUco marker detection."""
    dict_id: int  #: aruco dictionary identifier (e.g., 0 for cv2.aruco.DICT_4X4_50)
    marker_length: Optional[Union[float, Dict[int, float]]] = None  #: marker side length [m]
    params: cv2.aruco.DetectorParameters = field(default_factory=cv2.aruco.DetectorParameters)  #: aruco detection parameters

    @classmethod
    def from_dict(cls, data):
        config = super().from_dict(data)
        config.params = fix_aruco_params(config.params)
        return config


def _fix_aruco_axes(pose):
    rot_mat = pose.getRotationMatrix()
    rot_mat = np.column_stack((rot_mat[:, 1], -rot_mat[:, 0], rot_mat[:, 2]))
    return m3d.MatrixTransform(pose.getTranslation(), rot_mat)


def detect_aruco(img: np.ndarray, cfg: ArucoConfig, intrinsics=None):
    # detections
    aruco_dict = cv2.aruco.getPredefinedDictionary(cfg.dict_id)
    aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=cfg.params)
    if aruco_ids is None:
        return []

    # convert to marker detections
    detections = [
        MarkerDetection(corners=corners[0, :, :], identifier=aruco_id)
        for corners, aruco_id in zip(aruco_corners, aruco_ids.flatten())
    ]

    # convert to poses if intrinsics are available
    if cfg.marker_length is not None and intrinsics is not None:
        # marker points base
        object_pts_base = np.array([
            [-1, 1, 0],
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
        ])

        # iterate detections
        for det_idx, detection in enumerate(detections):
            # get object points for detection id
            if isinstance(cfg.marker_length, dict):
                if detection.identifier in cfg.marker_length:
                    ml = cfg.marker_length[detection.identifier]
                elif -1 in cfg.marker_length:
                    ml = cfg.marker_length[-1]
                else:
                    continue
            else:
                ml = cfg.marker_length
            object_pts = object_pts_base * (ml / 2)

            # detect with 2 possible solutions
            ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_pts, detection.corners,
                                                       intrinsics.camera_matrix, intrinsics.dist_coeffs,
                                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
            pose1 = m3d.MatrixTransform(tvecs[0].flatten(), cv2.Rodrigues(rvecs[0].flatten())[0])
            pose2 = m3d.MatrixTransform(tvecs[1].flatten(), cv2.Rodrigues(rvecs[1].flatten())[0])

            pose1 = _fix_aruco_axes(pose1)
            pose2 = _fix_aruco_axes(pose2)

            detections[det_idx].pose = pose1
            detections[det_idx].pose_options = [pose1, pose2]

    return detections
