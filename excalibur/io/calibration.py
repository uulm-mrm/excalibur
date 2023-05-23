import os.path as osp
from typing import Any, Dict
import yaml

import motion3d as m3d


def load_calibration(filename: str) -> Dict[Any, m3d.TransformInterface]:
    # check file
    if not osp.exists(filename):
        return None

    # load data
    with open(filename, 'r') as stream:
        data = yaml.safe_load(stream)

    # iterate data
    calib = {}
    for k, v in data.items():
        ttype = m3d.TransformType.FromChar(v['type'])
        calib[k] = m3d.TransformInterface.Factory(ttype, v['data'], unsafe=True).normalized_()

    return calib
