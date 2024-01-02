from pathlib import Path
from typing import Union

from .utils import load_yaml

import motion3d as m3d


def load_transforms(filename, normalize=True):
    # load data
    data = load_yaml(filename)

    # iterate data
    calib = {}
    for k, v in data.items():
        ttype = m3d.TransformType.FromChar(v['type'])
        calib[k] = m3d.TransformInterface.Factory(ttype, v['data'], unsafe=True)
        if normalize:
            calib[k].normalized_()

    return calib


def store_transform_container(filename: Union[Path, str], container: m3d.TransformContainer,
                              ttype: m3d.TransformType = m3d.TransformType.kMatrix) -> bool:
    # create output dir
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # save data
    motion_data = m3d.MotionData(ttype, container)
    status = m3d.M3DWriter.write(str(filename), motion_data, m3d.M3DFileType.kBinary)
    if status != m3d.M3DIOStatus.kSuccess:
        print(f"Error while writing: {m3d.M3DIOStatus(status)} ({status})")
        return False
    return True
