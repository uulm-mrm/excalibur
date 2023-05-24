#!/usr/bin/env python3
import argparse

import motion3d as m3d
import numpy as np
import scipy.io

from excalibur.utils.parsing import dir_path, file_path


MAT_VAR_A = 'BasePoseIntcpCoords'
MAT_VAR_B = 'gridPoseInCameraCoords'

MAT_VAR_WORLD = 'Worldpts'
MAT_VAR_IMG = 'imgPts'
MAT_VAR_K = 'KM'


def matlab2m3d(data):
    container = m3d.TransformContainer(has_stamps=False, has_poses=True)
    for i in range(data.shape[2]):
        t = m3d.MatrixTransform(data[:, :, i]).normalized()
        container.insert(t)
    return container


def main():
    # input arguments
    parser = argparse.ArgumentParser("Convert Ali HERW dataset from Matlab export.")
    parser.add_argument('input_file', type=file_path, help="Input filename (*.mat)")
    parser.add_argument('output_dir', type=dir_path, help="Output directory")
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    # check arguments
    if not input_file.is_file():
        raise RuntimeError(f"Input file does not exist or is not a file: {input_file}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist or is not a directory: {output_dir}")

    # read mat file
    mat_data = scipy.io.loadmat(input_file)

    # convert transforms
    container_a = matlab2m3d(mat_data[MAT_VAR_A])
    container_b = matlab2m3d(mat_data[MAT_VAR_B])

    # create and write motion data
    data_a = m3d.MotionData(m3d.TransformType.kMatrix, container_a)
    data_b = m3d.MotionData(m3d.TransformType.kMatrix, container_b)
    m3d.M3DWriter.writeBinary(str(output_dir / 'a.m3d'), data_a)
    m3d.M3DWriter.writeBinary(str(output_dir / 'b.m3d'), data_b)

    # read detections and camera matrix
    world_pts = mat_data[MAT_VAR_WORLD]
    img_pts = mat_data[MAT_VAR_IMG]
    Kmat = mat_data[MAT_VAR_K]

    # store detections
    detections = {
        'world_pts': world_pts,
        'img_pts': img_pts,
        'K': Kmat,
    }
    np.save(str(output_dir / 'detections.npy'), detections)


if __name__ == '__main__':
    main()
