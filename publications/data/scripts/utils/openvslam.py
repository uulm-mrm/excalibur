from enum import Enum
import os.path as osp

import motion3d as m3d
import numpy as np


class TrajectoryType(Enum):
    FRAME = 'frame_trajectory.txt'
    KEYFRAME = 'keyframe_trajectory.txt'


def load_transforms(export_dir, trajectory_type):
    # load data
    poses_data = np.loadtxt(osp.join(export_dir, trajectory_type.value))

    # create container
    tdata = m3d.TransformContainer(has_stamps=True, has_poses=True)
    for row in range(poses_data.shape[0]):
        # convert row
        timestamp = m3d.Time.FromSec(poses_data[row, 0])
        transform = m3d.MatrixTransform(poses_data[row, 1:], unsafe=True).normalized_()

        # left-hand to right-hand coordinate system
        t = np.diag([1, 1, -1, 1])
        matrix = np.linalg.inv(t) @ transform.asType(m3d.TransformType.kMatrix).getMatrix() @ t
        transform = m3d.MatrixTransform(matrix, unsafe=True).normalized_()

        # add to container
        tdata.insert(timestamp, transform)

    return tdata, m3d.TransformType.kMatrix
