#!/usr/bin/env python3
import argparse
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np

import excalibur.calibration
from excalibur.processing import ApplyToList, Compose, io as io_proc, transforms as transform_proc
from excalibur.io.calibration import CalibrationManager
from excalibur.io.utils import load_yaml
import excalibur.utils.evaluation as ev


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def calibrate(method_cfg, poses_a, poses_b):
    # initialize method
    method = method_cfg.create(excalibur.calibration.HERWCalibrationBase)
    method.set_transforms(poses_a, poses_b)

    # calibrate
    result = method.calibrate()

    # check success
    if not result.success:
        print(f"\nerror: {result.message}")
        return None, {}

    # metrics
    metrics = ev.get_metrics_multi(result)
    return result.calib, metrics


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Hand-Eye Robot-World Calibration")
    parser.add_argument('config', type=str, help="method configuration (*.yaml)")
    parser.add_argument('--a', type=str, nargs='+', required=True, help="files with poses of frame a (*.m3d)")
    parser.add_argument('--b', type=str, nargs='+', required=True, help="files with poses of frame b (*.m3d)")
    parser.add_argument('--norm-cfg', type=str, help="translation norm prior config (*.yaml)")
    parser.add_argument('--output', type=str, help="transformation output (*.yaml)")
    parser.add_argument('--novis', action='store_true', help="Skip visualization plots")
    args = parser.parse_args()

    # configuration
    method = ev.MethodConfig.load(args.config)
    if args.norm_cfg is not None:
        norm_config = load_yaml(args.norm_cfg)

        if 't_norm_x' in norm_config:
            t_norm_x = norm_config['t_norm_x']
            del norm_config['t_norm_x']
            if isinstance(t_norm_x, list):
                t_norm_x = float(np.linalg.norm(t_norm_x))
        else:
            t_norm_x = None

        if 't_norm_y' in norm_config:
            t_norm_y = norm_config['t_norm_y']
            del norm_config['t_norm_y']
            if isinstance(t_norm_y, list):
                t_norm_y = float(np.linalg.norm(t_norm_y))
        else:
            t_norm_y = None

        if t_norm_x is not None or t_norm_y is not None:
            norm_config['t_norms'] = [t_norm_x, t_norm_y]

        method.calib_kwargs.update(norm_config)

    # additional configuration
    sync_time_sec = method.get_extra_kwarg('sync')
    sync_time_nsec = int(sync_time_sec * 1e9) if sync_time_sec is not None else None

    print("== Config ==")
    method.print()

    # input files
    filenames = ev.combine_filenames(list_a=args.a, list_b=args.b)
    if filenames is None:
        sys.exit(-1)

    print("\n== Files ==")
    for a, b in filenames:
        print(f"- {a}, {b}")

    # load data
    print("\n== Calibration ==")
    transforms, _, _ = io_proc.load_transforms(filenames, return_frames=True)

    # processing
    pair_proc = Compose([
        transform_proc.MatchPoses(max_diff_nsec=sync_time_nsec),
        transform_proc.Normalized(),
    ])

    # build full processing chain
    proc = Compose([
        ApplyToList(pair_proc),
        transform_proc.MergeTransforms(),
    ])

    # apply processing chain
    poses_a, poses_b = proc(transforms)

    # print processed motions
    print(f"Number of synchronous samples: {len(poses_a)}")
    if len(poses_a) < 3:
        print("\nerror: cannot calibrate with less than 3 samples")
        sys.exit(-1)

    # calibrate
    print("Calibrate")
    calib, metrics = calibrate(method, poses_a, poses_b)
    if calib is None:
        sys.exit(-1)

    # print result
    print("\n== Result ==")
    print("-- X --")
    ev.print_calib(calib.x, norm=True)
    print("-- Y --")
    ev.print_calib(calib.y, norm=True)

    print("\n== Metrics ==")
    ev.print_metrics(metrics)

    # export calibration
    if args.output is not None:
        print(f"\nExport calibration to '{args.output}'")
        manager = CalibrationManager()
        manager.add('x_from', 'x_to', calib.x)
        manager.add('y_from', 'y_to', calib.y)
        manager.save(args.output)

    # visualization
    if not args.novis and 'errors' in metrics and metrics['errors'] is not None:
        ev.plot_ransac_errors(metrics['errors'], method, log=True)
        plt.show()


if __name__ == '__main__':
    main()
