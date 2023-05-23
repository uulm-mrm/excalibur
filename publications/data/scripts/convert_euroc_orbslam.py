#!/usr/bin/env python3
import argparse
import os.path as osp
from pathlib import Path
import sys

import motion3d as m3d

from utils.euroc import EurocFrame, load_euroc_calib, SEQUENCES
from utils.orbslam import load_transforms, TrajectoryType
from utils.parse import ParseEnum


def convert_euroc_orbslam(orbslam_dir, euroc_root_dir, output_dir, frame, binary):
    # iterate sequences
    for seq in SEQUENCES:
        # sequence directory
        euroc_seq_dir = osp.join(euroc_root_dir, seq, 'mav0')

        # load data
        tdata, ttype = load_transforms(osp.join(orbslam_dir, seq), TrajectoryType.KEYFRAME)
        calib = load_euroc_calib(euroc_seq_dir, frame)

        # create motion data
        motion_data = m3d.MotionData(ttype, tdata, calib, frame.value)

        # write and check
        Path(output_dir).mkdir(parents=False, exist_ok=True)
        output_filename = osp.join(output_dir, f'{seq}.m3d')
        file_type = m3d.M3DFileType.kBinary if binary else m3d.M3DFileType.kASCII
        status = m3d.M3DWriter.write(output_filename, motion_data, file_type)
        if status != m3d.M3DIOStatus.kSuccess:
            print(f"Error while writing: {m3d.M3DIOStatus(status)} ({status})")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert ORB-SLAM data of EuRoC in TUM format.")
    parser.add_argument('orbslam_dir', type=str, help="ORB-SLAM directory")
    parser.add_argument('euroc_dir', type=str, help="EuRoC directory")
    parser.add_argument('output_dir', type=str, help="Output directory")
    parser.add_argument('--frame', required=True, action=ParseEnum, enum_type=EurocFrame)
    parser.add_argument('--binary', action='store_true', help="Use binary m3d format")
    args = parser.parse_args()

    convert_euroc_orbslam(args.orbslam_dir, args.euroc_dir, args.output_dir, args.frame, args.binary)


if __name__ == '__main__':
    main()
