from enum import auto, Enum

import motion3d as m3d
import numpy as np


SEQUENCES = range(11)


class KittiFrame(Enum):
    CAM0 = 'cam0'
    CAM1 = 'cam1'
    CAM2 = 'cam2'
    CAM3 = 'cam3'
    VELO = 'velodyne'


def _read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.

    Note: The implementation is adopted from pykitti.utils.read_calib_file():
          https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_kitti_odometry_calib(calib_filename, frame=None):
    """
    Load and compute intrinsic and extrinsic calibration parameters.

    Note: The implementation is adopted from pykitti.odometry._get_file_lists():
          https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
    """
    # Load the calibration file
    filedata = _read_calib_file(calib_filename)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    # T_cam0_velo is the frame transformation cam0 -> velodyne
    T_cam0_velo = np.reshape(filedata['Tr'], (3, 4))
    T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
    T_cam1_velo = T1.dot(T_cam0_velo)
    T_cam2_velo = T2.dot(T_cam0_velo)
    T_cam3_velo = T3.dot(T_cam0_velo)

    # Select coordinate frame
    if frame == KittiFrame.CAM0:
        return m3d.MatrixTransform(T_cam0_velo, unsafe=True).normalized_().inverse_()
    elif frame == KittiFrame.CAM1:
        return m3d.MatrixTransform(T_cam1_velo, unsafe=True).normalized_().inverse_()
    elif frame == KittiFrame.CAM2:
        return m3d.MatrixTransform(T_cam2_velo, unsafe=True).normalized_().inverse_()
    elif frame == KittiFrame.CAM3:
        return m3d.MatrixTransform(T_cam3_velo, unsafe=True).normalized_().inverse_()
    elif frame == KittiFrame.VELO:
        return m3d.MatrixTransform()
    else:
        raise NotImplementedError(f"Coordinate frame '{frame}' not implemented")
