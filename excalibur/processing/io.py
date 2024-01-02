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


def load_transform_pair(filename1, filename2, normalized=True, return_frames=False):
    # read files
    data1, status1 = m3d.M3DReader.read(str(filename1))
    data2, status2 = m3d.M3DReader.read(str(filename2))

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

    # get transforms
    transforms1 = data1.getTransforms()
    transforms2 = data2.getTransforms()
    if normalized:
        transforms1.normalized_()
        transforms2.normalized_()

    if return_frames:
        return transforms1, transforms2, calib, data1.getFrameId(), data2.getFrameId()
    else:
        return transforms1, transforms2, calib


def load_transforms(filenames, normalized=True, return_frames=False):
    data_list = []
    frames_out = None
    calib_out = None

    for filename1, filename2 in filenames:
        # load data
        if return_frames:
            data1, data2, calib, frame1, frame2 = load_transform_pair(
                filename1, filename2, normalized=normalized, return_frames=True)

            if frames_out is None:
                frames_out = (frame1, frame2)
            elif frames_out[0] != frame1 or frames_out[1] != frame2:
                raise RuntimeError("Frame ids of transform pairs are not identical")
        else:
            data1, data2, calib = load_transform_pair(
                filename1, filename2, normalized=normalized, return_frames=False)

        # check calib
        if calib_out is None:
            calib_out = calib
        else:
            if not calib.isEqual(calib_out):
                raise RuntimeError("Calibrations of transform pairs are not identical")

        # store data
        data_list.append((data1, data2))

    if return_frames:
        return data_list, calib_out, frames_out
    else:
        return data_list, calib_out
