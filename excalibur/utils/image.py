from typing import List, Tuple

import cv2
import motion3d as m3d
import numpy as np

from excalibur.io.camera import CameraIntrinsics
from excalibur.io.geometry import Plane


def project_image_point_to_plane(point: np.ndarray, plane: Plane, intrinsics: CameraIntrinsics) -> np.ndarray:
    # check intrinsics
    if np.sum(np.abs(intrinsics.dist_coeffs)) != 0.0:
        raise NotImplementedError("Projection not implemented for distorted images")

    # line in camera coordinates
    line_vec = np.linalg.inv(intrinsics.camera_matrix) @ np.vstack([point.reshape(2, 1), 1.0])

    # intersection with ground plane
    point_dist = plane.distance / np.dot(plane.normal, line_vec)[0]
    return line_vec.squeeze() * point_dist


def project_cc_to_ic(points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points_ic, _ = cv2.projectPoints(points, rvec=np.zeros(3), tvec=np.zeros(3),
                                     cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs)
    return points_ic.squeeze()


def _cv2_to_m3d(rvec: np.ndarray, tvec: np.ndarray) -> m3d.TransformInterface:
    return m3d.MatrixTransform(tvec.flatten(), cv2.Rodrigues(rvec.flatten())[0])


def solve_pnp(object_pts: np.ndarray, image_pts: np.ndarray, intrinsics: CameraIntrinsics,
              flags: int = cv2.SOLVEPNP_ITERATIVE) -> m3d.TransformInterface:
    retval, rvec, tvec = cv2.solvePnP(
        object_pts, image_pts, cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs, flags=flags)
    return _cv2_to_m3d(rvec, tvec)


def solve_pnp_generic(object_pts: np.ndarray, image_pts: np.ndarray, intrinsics: CameraIntrinsics,
                      flags: int = cv2.SOLVEPNP_ITERATIVE) -> Tuple[List[m3d.TransformInterface], np.ndarray]:
    ret, rvecs, tvecs, reproj_err = cv2.solvePnPGeneric(
        object_pts, image_pts, cameraMatrix=intrinsics.camera_matrix,distCoeffs=intrinsics.dist_coeffs, flags=flags)
    transforms = [_cv2_to_m3d(rvec, tvec) for rvec, tvec in zip(rvecs, tvecs)]
    return transforms, reproj_err
