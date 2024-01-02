#!/usr/bin/env python3
import argparse
import sys

import motion3d as m3d

from excalibur.io.calibration import CalibrationManager
from excalibur.utils.parsing import ParsePybindEnum


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Apply transform operation and/or store in ascii or binary format.")
    parser.add_argument('input', type=str, help="input file (*.m3d)")
    parser.add_argument('output', type=str, help="output file (*.m3d)")
    parser.add_argument('--change', type=str, nargs=3, metavar=("calibration_file", "from", "to"), help="change frame")
    parser.add_argument('--inverse', action='store_true', help="invert transformations")
    parser.add_argument('--ascii', action='store_true', help="use ascii instead of binary output format")
    parser.add_argument('--precision', type=int, default=16, help="ascii number precision")
    parser.add_argument('--type', action=ParsePybindEnum, enum_type=m3d.TransformType, help="change transform type")
    args = parser.parse_args()

    # check input
    if args.change is not None and args.inverse:
        parser.error("Operations 'change' and 'inverse' are not allowed at the same time.")

    # read
    print(f"Read '{args.input}'")
    motion_data, status = m3d.M3DReader.read(args.input)

    # check
    if status != m3d.M3DIOStatus.kSuccess:
        print(f"Error: could not read file: {status}")
        sys.exit(-1)

    # apply operation
    transforms = motion_data.getTransforms()

    if args.change is not None:
        try:
            manager = CalibrationManager.load(args.change[0])
        except FileNotFoundError:
            print(f"Error: calibration file '{args.change[0]}' not found")
            sys.exit(-1)

        calib = manager.get(args.change[1], args.change[2])
        if calib is None:
            print(f"Error: transformation from '{args.change[1]}' to '{args.change[2]}' "
                  f"not found in '{args.change[0]}'")
            sys.exit(-1)

        transforms.changeFrame_(calib)

    elif args.inverse:
        transforms.inverse_()

    motion_data.setTransforms(transforms)

    # convert type
    if args.type is not None:
        motion_data.setTransformType(args.type)

    # write
    if args.ascii:
        print(f"Write '{args.output}' in ascii format")
        m3d.M3DWriter.writeASCII(args.output, motion_data, precision=args.precision)
    else:
        print(f"Write '{args.output}' in binary format")
        m3d.M3DWriter.writeBinary(args.output, motion_data)


if __name__ == '__main__':
    main()
