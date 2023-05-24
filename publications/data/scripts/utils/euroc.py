from enum import Enum
import os.path as osp

import motion3d as m3d
import numpy as np
import yaml


SEQUENCES = [
    'MH_01_easy',
    'MH_02_easy',
    'MH_03_medium',
    'MH_04_difficult',
    'MH_05_difficult',
    'V1_01_easy',
    'V1_02_medium',
    'V1_03_difficult',
    'V2_01_easy',
    'V2_02_medium',
    'V2_03_difficult'
]


class EurocFrame(Enum):
    CAM0 = 'cam0'
    CAM1 = 'cam1'
    GROUND_TRUTH = 'state_groundtruth_estimate0'


def load_euroc_calib(sequence_dir, frame):
    # load calib file
    sensor_file = osp.join(sequence_dir, frame.value, 'sensor.yaml')
    with open(sensor_file, 'r') as stream:
        data = yaml.safe_load(stream)

    # read calib
    matrix = np.array(data['T_BS']['data']).reshape(4, 4)

    # convert
    return m3d.MatrixTransform(matrix, unsafe=True).normalized()
