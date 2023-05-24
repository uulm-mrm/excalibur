#!/usr/bin/env python3
import argparse
import os.path as osp
from pathlib import Path
import sys

import numpy as np

import motion3d as m3d

from utils.kitti import KittiFrame, load_kitti_odometry_calib, SEQUENCES


def convert_kitti_aloam(aloam_dir, kitti_root_dir, output_dir, binary):
    # iterate sequences
    for seq in SEQUENCES:
        # sequence directory
        seq_str = f'{seq:02}'
        kitti_seq_dir = osp.join(kitti_root_dir, 'sequences', seq_str)

        # load data
        poses_data = np.loadtxt(osp.join(aloam_dir, f'{seq_str}.txt'))
        times_data = np.loadtxt(osp.join(kitti_seq_dir, 'times.txt'))
        calib = load_kitti_odometry_calib(osp.join(kitti_seq_dir, 'calib.txt'), KittiFrame.VELO)

        # create container
        tdata = m3d.TransformContainer(has_stamps=True, has_poses=True)
        for row in range(poses_data.shape[0]):
            timestamp = m3d.Time.FromNSec(np.uint64(times_data[row] * 1e9))
            transform = m3d.QuaternionTransform(poses_data[row, :], unsafe=True).normalized_()
            tdata.insert(timestamp, transform)

        # create motion data
        motion_data = m3d.MotionData(m3d.TransformType.kQuaternion, tdata, calib, KittiFrame.VELO.value)

        # write and check
        Path(output_dir).mkdir(parents=False, exist_ok=True)
        output_filename = osp.join(output_dir, f'{seq_str}.m3d')
        file_type = m3d.M3DFileType.kBinary if binary else m3d.M3DFileType.kASCII
        status = m3d.M3DWriter.write(output_filename, motion_data, file_type)
        if status != m3d.M3DIOStatus.kSuccess:
            print(f"Error while writing: {m3d.M3DIOStatus(status)} ({status})")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert A-LOAM data of KITTI odometry in KITTI format.")
    parser.add_argument('aloam_dir', type=str, help="A-LOAM directory")
    parser.add_argument('kitti_dir', type=str, help="KITTI directory")
    parser.add_argument('output_dir', type=str, help="Output directory")
    parser.add_argument('--binary', action='store_true', help="Use binary m3d format")
    args = parser.parse_args()

    convert_kitti_aloam(args.aloam_dir, args.kitti_dir, args.output_dir, args.binary)


if __name__ == '__main__':
    main()
