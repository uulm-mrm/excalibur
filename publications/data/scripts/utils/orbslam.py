from enum import Enum
import os.path as osp

import motion3d as m3d
import numpy as np


class TrajectoryType(Enum):
    CAMERA = 'CameraTrajectory.txt'
    KEYFRAME = 'KeyFrameTrajectory.txt'


def load_transforms(export_dir, trajectory_type):
    # load data
    poses_data = np.loadtxt(osp.join(export_dir, trajectory_type.value))

    # create container
    tdata = m3d.TransformContainer(has_stamps=True, has_poses=True)
    for row in range(poses_data.shape[0]):
        # stamp
        timestamp = m3d.Time.FromNSec(np.int64(poses_data[row, 0]))

        # transform
        translation = poses_data[row, 1:4]
        quaternion = m3d.Quaternion(x=poses_data[row, 4], y=poses_data[row, 5],
                                    z=poses_data[row, 6], w=poses_data[row, 7])
        transform = m3d.QuaternionTransform(translation, quaternion, unsafe=True).normalized_()

        # insert
        tdata.insert(timestamp, transform)

    return tdata, m3d.TransformType.kQuaternion
