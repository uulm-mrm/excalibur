from dataclasses import dataclass
from typing import Optional, Tuple

from checkerboard import detect_checkerboard as detect_checkerboard_lib
import cv2
import motion3d as m3d
import numpy as np

from excalibur.io.dataclass import DataclassIO

from .utils import MarkerDetection


def detect_checkerboard_corners(img, board_dim, flags=None, fast_check=False):
    # default flags
    if flags is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # fast check
    if fast_check:
        pattern_found, _ = cv2.findChessboardCorners(
            img, board_dim, flags=flags + cv2.CALIB_CB_FAST_CHECK)
        if not pattern_found:
            return None

    # opencv detector
    try:
        pattern_found, corners = cv2.findChessboardCorners(img, board_dim, flags=flags)
    except Exception:
        pattern_found = False

    # checkerboard detector
    if not pattern_found:
        corners, _ = detect_checkerboard_lib(img, (board_dim[1], board_dim[0]), winsize=5)
        pattern_found = corners is not None
        if pattern_found:
            corners = np.ascontiguousarray(corners.astype(np.float32))  # for opencv subpixel accuracy

    # subpixel accuracy
    if not pattern_found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)
    return corners.squeeze()


def generate_checkerboard_world_points(board_dim, square_size, matlab=False):
    if matlab:
        object_pts = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
        object_pts[:, :2] = np.mgrid[0:board_dim[1], 0:board_dim[0]].T.reshape(-1, 2)
        object_pts *= square_size

        object_pts = np.column_stack([object_pts[:, 1], object_pts[:, 0], object_pts[:, 2]])

        object_pts -= np.array([(board_dim[0] - 1) * square_size / 2,
                                (board_dim[1] - 1) * square_size / 2, 0])
        object_pts[:, 1] *= -1
    else:
        object_pts = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
        object_pts[:, :2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2)
        object_pts *= square_size

        object_pts -= np.array([(board_dim[0] - 1) * square_size / 2,
                                (board_dim[1] - 1) * square_size / 2, 0])
        object_pts[:, 1] *= -1
    return object_pts


_CHECKERBOARD_FIX = m3d.EulerTransform([0, 0, 0], [np.pi, 0, 0], m3d.EulerAxes.kSXYZ)


def _fix_checkerboard_pose(pose):
    # check if checkerboard z-axis is pointing towards camera
    pose_mat = pose.asType(m3d.TransformType.kMatrix)
    z_axis = pose_mat.getRotationMatrix()[:, 2]
    t_vec = pose_mat.getTranslation()

    if z_axis @ t_vec > 0:
        # z-axis and translation vector are pointing in the same direction -> pose must be fixed
        pose *= _CHECKERBOARD_FIX

    return pose


def estimate_checkerboard_pose(corners, board_dim, square_size, intrinsics, matlab=False):
    # world points
    object_pts = generate_checkerboard_world_points(board_dim, square_size, matlab=matlab)

    # get pose
    ret, rvec, tvec = cv2.solvePnP(object_pts, corners, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    if rvec is None or tvec is None:
        return None

    pose = m3d.MatrixTransform(tvec.flatten(), cv2.Rodrigues(rvec.flatten())[0])
    return _fix_checkerboard_pose(pose)


def estimate_checkerboard_pose_multi(corners, board_dim, square_size, intrinsics, matlab=False):
    # world points
    object_pts = generate_checkerboard_world_points(board_dim, square_size, matlab=matlab)

    # detect with 2 possible solutions
    ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_pts, corners,
                                               intrinsics.camera_matrix, intrinsics.dist_coeffs,
                                               flags=cv2.SOLVEPNP_IPPE)
    pose1 = m3d.MatrixTransform(tvecs[0].flatten(), cv2.Rodrigues(rvecs[0].flatten())[0])
    pose2 = m3d.MatrixTransform(tvecs[1].flatten(), cv2.Rodrigues(rvecs[1].flatten())[0])
    return _fix_checkerboard_pose(pose1), _fix_checkerboard_pose(pose2)


@dataclass
class CheckerboardConfig(DataclassIO):
    """Configuration for checkerboard detection."""
    board_dim: Tuple[int, int]  #: number of cells (length/x, width/y)
    square_length: Optional[float] = None  #: cell size [m]
    flags: Optional[int] = None  #: checkerboard detection flags for cv2.findChessboardCorners
    fast_check: bool = False  #: run fast check before full estimation


def detect_checkerboard(img: np.ndarray, cfg: CheckerboardConfig, intrinsics=None):
    # corners
    corners = detect_checkerboard_corners(img, cfg.board_dim, flags=cfg.flags, fast_check=cfg.fast_check)
    if corners is None:
        return None
    detection = MarkerDetection(corners=corners)

    # pose
    if cfg.square_length is not None and intrinsics is not None:
        detection.pose_options = estimate_checkerboard_pose_multi(corners, cfg.board_dim, cfg.square_length, intrinsics)
        detection.pose = detection.pose_options[0]

    return detection
