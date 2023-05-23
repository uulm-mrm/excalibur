#!/usr/bin/env python3
import argparse
import os.path as osp
from pathlib import Path
import sys

import numpy as np

import motion3d as m3d

from utils.euroc import EurocFrame, load_euroc_calib, SEQUENCES


def convert_euroc_ground_truth(euroc_root_dir, output_dir, binary):
    # iterate sequences
    for seq in SEQUENCES:
        # ground-truth directory
        euroc_seq_dir = osp.join(euroc_root_dir, seq, 'mav0')

        # load data
        poses_data = np.loadtxt(osp.join(euroc_seq_dir, EurocFrame.GROUND_TRUTH.value, 'data.csv'), delimiter=',',
                                dtype=np.dtype('int,float,float,float,float,float,float,float,'))
        calib = load_euroc_calib(euroc_seq_dir, EurocFrame.GROUND_TRUTH)

        # create container
        tdata = m3d.TransformContainer(has_stamps=True, has_poses=True)
        for row in range(poses_data.shape[0]):
            # convert row
            timestamp = m3d.Time.FromNSec(poses_data[row][0])
            translation = np.array([poses_data[row][1], poses_data[row][2], poses_data[row][3]])
            quaternion = m3d.Quaternion(w=poses_data[row][4], x=poses_data[row][5],
                                        y=poses_data[row][6], z=poses_data[row][7])
            transform = m3d.QuaternionTransform(translation, quaternion, unsafe=True).normalized_()

            tdata.insert(timestamp, transform)

        # create motion data
        motion_data = m3d.MotionData(m3d.TransformType.kQuaternion, tdata, calib, 'body')

        # write and check
        Path(output_dir).mkdir(parents=False, exist_ok=True)
        output_filename = osp.join(output_dir, f'{seq}.m3d')
        file_type = m3d.M3DFileType.kBinary if binary else m3d.M3DFileType.kASCII
        status = m3d.M3DWriter.write(output_filename, motion_data, file_type)
        if status != m3d.M3DIOStatus.kSuccess:
            print(f"Error while writing: {m3d.M3DIOStatus(status)} ({status})")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert ground-truth data of EuRoC.")
    parser.add_argument('euroc_dir', type=str, help="EuRoC directory")
    parser.add_argument('output_dir', type=str, help="Output directory")
    parser.add_argument('--binary', action='store_true', help="Use binary m3d format")
    args = parser.parse_args()

    convert_euroc_ground_truth(args.euroc_dir, args.output_dir, args.binary)


if __name__ == '__main__':
    main()
