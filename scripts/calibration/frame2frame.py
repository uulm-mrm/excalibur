#!/usr/bin/env python3
import argparse
import signal
import sys

import matplotlib.pyplot as plt

import excalibur.calibration
from excalibur.processing import ApplyToList, Compose, io as io_proc, transforms as transform_proc
from excalibur.io.calibration import CalibrationManager
import excalibur.utils.evaluation as ev


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def calibrate(method_cfg, poses_a, poses_b, calib_x):
    # initialize method
    method = method_cfg.create(excalibur.calibration.Frame2FrameCalibrationBase)
    method.set_transforms(poses_a, poses_b)

    # calibrate
    result = method.calibrate()

    # check success
    if not result.success:
        print(f"\nerror: {result.message}")
        return None, {}

    # metrics
    metrics = ev.get_metrics_single(result, calib_x)
    return result.calib, metrics


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Frame2Frame Calibration")
    parser.add_argument('config', type=str, help="method configuration (*.yaml)")
    parser.add_argument('--a', type=str, nargs='+', required=True, help="files with poses of sensor a (*.m3d)")
    parser.add_argument('--b', type=str, nargs='+', required=True, help="files with poses of sensor b (*.m3d)")
    parser.add_argument('--frame-a', type=str, help="frame identifier for sensor a")
    parser.add_argument('--frame-b', type=str, help="frame identifier for sensor b")
    parser.add_argument('--output', type=str, help="transformation output (*.yaml)")
    parser.add_argument('--novis', action='store_true', help="Skip visualization plots")
    args = parser.parse_args()

    # configuration
    method = ev.MethodConfig.load(args.config)
    print("== Config ==")
    method.print()

    # additional configuration
    sync_time_sec = method.get_extra_kwarg('sync')
    sync_time_nsec = int(sync_time_sec * 1e9) if sync_time_sec is not None else None
    transform_filter = method.get_extra_kwarg('transform_filter')

    # input files
    filenames = ev.combine_filenames(list_a=args.a, list_b=args.b)
    if filenames is None:
        sys.exit(-1)

    print("\n== Files ==")
    for a, b in filenames:
        print(f"- {a}, {b}")

    # load data
    print("\n== Calibration ==")
    transforms, calib_gt, frames = io_proc.load_transforms(filenames, return_frames=True)
    frame_a, frame_b = ev.get_frames(filenames[0][0], filenames[0][1], frames, (args.frame_a, args.frame_b))

    # show ground truth
    if calib_gt is None:
        print("No ground truth available")
    else:
        print("Ground truth available:")
        ev.print_calib(calib_gt)

    # processing
    pair_proc = Compose([
        transform_proc.MatchPoses(max_diff_nsec=sync_time_nsec),
        transform_proc.Normalized(),
    ])
    if transform_filter is not None:
        pair_proc.append(transform_proc.TransformFilter(**transform_filter))

    proc = Compose([
        ApplyToList(pair_proc),
        transform_proc.MergeTransforms()
    ])

    poses_a, poses_b = proc(transforms)
    print(f"Number of synchronous samples: {len(poses_a)}")
    if len(poses_a) < 1:
        print("\nerror: cannot calibrate with less than 1 samples")
        sys.exit(-1)

    # calibrate
    print("Calibrate")
    calib, metrics = calibrate(method, poses_a, poses_b, calib_gt)
    if calib is None:
        sys.exit(-1)

    # print result
    print("\n== Result ==")
    print(f"Frames:           {frame_a} -> {frame_b}")
    ev.print_calib(calib)

    print("\n== Metrics ==")
    ev.print_metrics(metrics)

    # export calibration
    if args.output is not None:
        print(f"\nExport calibration to '{args.output}'")
        manager = CalibrationManager()
        manager.add(frame_a, frame_b, calib)
        manager.save(args.output)

    # plot sample errors
    if not args.novis and 'errors' in metrics and metrics['errors'] is not None:
        ev.plot_ransac_errors(metrics['errors'], method, log=True)
        plt.show()


if __name__ == '__main__':
    main()
