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
