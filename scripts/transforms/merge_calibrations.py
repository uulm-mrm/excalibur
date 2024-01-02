#!/usr/bin/env python3
import argparse
import signal

import matplotlib.pyplot as plt

from excalibur.io.calibration import CalibrationManager


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Merge multiple calibration files.")
    parser.add_argument('inputs', type=str, nargs='+', help="calibration files (*.yaml)")
    parser.add_argument('--output', type=str, help="output file (*.yaml)")
    parser.add_argument('--show', action='store_true', help="show merged transforms")
    args = parser.parse_args()

    # load base file
    print(f"Load {args.inputs[0]}")
    base_manager = CalibrationManager.load(args.inputs[0])

    # extend base file
    for filename in args.inputs[1:]:
        print(f"Load {filename}")
        sub_manager = CalibrationManager.load(filename)
        base_manager.extend(sub_manager)

    print(f"\nMerged collection contains in total {len(base_manager.frames())} frames:")
    for frame in base_manager.frames():
        print(f"- {frame}")

    # save merged transforms
    if args.output is not None:
        print(f"\nSave collection to '{args.output}'")
        base_manager.save(args.output)

    # show merged transforms
    if args.show and len(base_manager.frames()) > 0:
        base_manager.plot_frames(base_manager.frames()[0], length=0.2)
        plt.show()


if __name__ == '__main__':
    main()
