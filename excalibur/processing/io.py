import sys
import warnings

import motion3d as m3d


def load_transforms_file(filename, normalized=True):
    # read files
    data, status = m3d.M3DReader.read(filename)

    # check status
    if status != m3d.M3DIOStatus.kSuccess:
        raise RuntimeError(f"Could not read '{filename} - {status}'")

    # normalize
    if normalized:
        return data.getTransforms().normalized_()
    else:
        return data.getTransforms()


def load_transform_pair(filename1, filename2, normalized=True):
    # read files
    data1, status1 = m3d.M3DReader.read(filename1)
    data2, status2 = m3d.M3DReader.read(filename2)

    # check status
    if status1 != m3d.M3DIOStatus.kSuccess:
        raise RuntimeError(f"Could not read '{filename1} - {status1}'")
    if status2 != m3d.M3DIOStatus.kSuccess:
        raise RuntimeError(f"Could not read '{filename2} - {status2}'")

    # create calib
    if data1.getOrigin() is not None and data2.getOrigin() is not None:
        if normalized:
            calib = data2.getOrigin().normalized_() / data1.getOrigin().normalized_()
            calib.normalized_()
        else:
            calib = data2.getOrigin() / data1.getOrigin()
    else:
        calib = None

    # return transforms and calib
    if normalized:
        return data1.getTransforms().normalized_(), data2.getTransforms().normalized_(), calib
    else:
        return data1.getTransforms(), data2.getTransforms(), calib


def load_transforms(filenames, normalized=True):
    data_list = []
    calib_out = None

    for filename1, filename2 in filenames:
        # load data
        data1, data2, calib = load_transform_pair(filename1, filename2, normalized=normalized)

        # check calib
        if calib_out is None:
            calib_out = calib
        else:
            if not calib.isEqual(calib_out):
                raise RuntimeError("Calibrations of transform pairs are not identical")

        # store data
        data_list.append((data1, data2))

    return data_list, calib_out
