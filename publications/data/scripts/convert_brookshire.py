#!/usr/bin/env python3
import argparse
import sys

import numpy as np

import motion3d as m3d


def convert_brookshire(input_file, output_file, add_calib, binary):
    # load data
    data = np.loadtxt(input_file)

    # create container
    tdata = m3d.TransformContainer(has_stamps=False, has_poses=True)
    for row in range(data.shape[0]):
        transform = m3d.DualQuaternionTransform(data[row, :], unsafe=True).normalized_()
        tdata.insert(transform)

    # create motion data
    if add_calib:
        calib = m3d.EulerTransform(translation=np.array([-0.04, 0.025, 0]),
                                   angles=np.deg2rad([-4, 0, 180.0]),
                                   axes=m3d.EulerAxes.kSXYZ)
    else:
        calib = m3d.DualQuaternionTransform()

    motion_data = m3d.MotionData(m3d.TransformType.kDualQuaternion, tdata, calib)

    # write and check
    file_type = m3d.M3DFileType.kBinary if binary else m3d.M3DFileType.kASCII
    status = m3d.M3DWriter.write(output_file, motion_data, file_type)
    if status != m3d.M3DIOStatus.kSuccess:
        print(f"Error while writing: {m3d.M3DIOStatus(status)} ({status})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert Brookshire data.")
    parser.add_argument('input', type=str, help="Input file (*.txt)")
    parser.add_argument('output', type=str, help="Output file (*.m3d)")
    parser.add_argument('--calib', action='store_true', help="Add calibration as origin (for sensor2)")
    parser.add_argument('--binary', action='store_true', help="Use binary m3d format")
    args = parser.parse_args()

    convert_brookshire(args.input, args.output, args.calib, args.binary)


if __name__ == '__main__':
    main()
