#!/usr/bin/env python3
import argparse
import signal
import sys

import matplotlib.pyplot as plt

from excalibur.io.calibration import CalibrationManager


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Show frames in calibration file.")
    parser.add_argument('filename', type=str, help="calibration file (*.yaml)")
    parser.add_argument('origin', type=str, nargs='?', help="origin frame")
    parser.add_argument('--l', type=float, default=0.2, help="axes length")
    args = parser.parse_args()

    # load
    manager = CalibrationManager.load(args.filename)
    if len(manager.frames()) == 0:
        print(f"File '{args.filename}' does not contain any frames.")
        sys.exit(0)

    # origin frame
    origin = args.origin
    if origin is None:
        origin = manager.frames()[0]
    elif origin not in manager.frames():
        print(f"Error: origin frame '{origin}' not found.")
        sys.exit(-1)

    # show
    manager.plot_frames(origin, args.l)
    plt.show()


if __name__ == '__main__':
    main()
