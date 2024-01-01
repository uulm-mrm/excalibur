#!/usr/bin/env python3
import argparse
import signal
import sys

import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np

import excalibur.calibration
from excalibur.processing import ApplyToList, Compose, io as io_proc, transforms as transform_proc
from excalibur.io.calibration import CalibrationManager
from excalibur.io.utils import load_yaml
import excalibur.utils.evaluation as ev


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def calibrate(method_cfg, motions_a, motions_b, calib_x):
    # initialize method
    method = method_cfg.create(excalibur.calibration.HandEyeCalibrationBase)
    method.set_transforms(motions_a, motions_b)

    # calibrate
    result = method.calibrate()

    # check success
    if not result.success:
        print(f"\nerror: {result.message}")
        return None, {}

    # metrics
    metrics = ev.get_metrics_single(result, calib_x)
    return result.calib, metrics


def plot_rotation_angles(motions_a, motions_b, frame_a='a', frame_b='b'):
    # prepare angles
    motions_a_ax = motions_a.asType(m3d.TransformType.kAxisAngle)
    motions_b_ax = motions_b.asType(m3d.TransformType.kAxisAngle)
    rot_angles_a = np.array([m.getAngle() for m in motions_a_ax])
    rot_angles_b = np.array([m.getAngle() for m in motions_b_ax])

    # plot angles
    plt.figure()
    plt.subplot(211)
    plt.plot(np.rad2deg(rot_angles_a), color='tab:blue', label=frame_a)
    plt.plot(np.rad2deg(rot_angles_b), color='tab:orange', label=frame_b)
    plt.xlabel("Sample Index")
    plt.ylabel("Rot. Angle [deg]")
    plt.legend()

    # plot offsets
    plt.subplot(212)
    plt.plot(np.rad2deg(rot_angles_a - rot_angles_b), color='tab:red')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Rot. Angle Offset [deg]")


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration")
    parser.add_argument('config', type=str, help="method configuration (*.yaml)")
    parser.add_argument('--a', type=str, nargs='+', required=True, help="files with motions of sensor a (*.m3d)")
    parser.add_argument('--b', type=str, nargs='+', required=True, help="files with motions of sensor b (*.m3d)")
    parser.add_argument('--frame-a', type=str, help="frame identifier for sensor a")
    parser.add_argument('--frame-b', type=str, help="frame identifier for sensor b")
    parser.add_argument('--norm-cfg', type=str, help="translation norm prior config (*.yaml)")
    parser.add_argument('--output', type=str, help="transformation output (*.yaml)")
    parser.add_argument('--novis', action='store_true', help="Skip visualization plots")
    args = parser.parse_args()

    # configuration
    method = ev.MethodConfig.load(args.config)
    if args.norm_cfg is not None:
        norm_config = load_yaml(args.norm_cfg)
        if 't_norm' in norm_config and isinstance(norm_config['t_norm'], list):
            norm_config['t_norm'] = float(np.linalg.norm(norm_config['t_norm']))
        method.calib_kwargs.update(norm_config)

    # additional configuration
    sync_time_sec = method.get_extra_kwarg('sync')
    sync_time_nsec = int(sync_time_sec * 1e9) if sync_time_sec is not None else None
    min_step = method.get_extra_kwarg('min_step')
    max_step = method.get_extra_kwarg('max_step')
    max_rot_diff_deg = method.get_extra_kwarg('max_rot_diff_deg')
    transform_filter = method.get_extra_kwarg('transform_filter')
    scaled = method.get_extra_kwarg('scaled', default=False)

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
        transform_proc.PosesToMotions(min_step_sec=min_step, max_step_sec=max_step),
        transform_proc.Normalized(),
    ])
    if max_rot_diff_deg is not None:
        pair_proc.append(transform_proc.RemoveMotionOutliers(max_rotation_diff_deg=max_rot_diff_deg))
    if transform_filter is not None:
        pair_proc.append(transform_proc.TransformFilter(**transform_filter))

    # build full processing chain
    proc = Compose([
        ApplyToList(pair_proc),
    ])
    if scaled:
        proc.append(transform_proc.PivotTransforms())
    else:
        proc.append(transform_proc.MergeTransforms())

    # apply processing chain
    motions_a, motions_b = proc(transforms)

    # create containers with all motions
    if scaled:
        motions_a_acc = motions_a[0].copy()
        motions_b_acc = motions_b[0].copy()
        for m in motions_a[1:]:
            motions_a_acc.extend(m)
        for m in motions_b[1:]:
            motions_b_acc.extend(m)
    else:
        motions_a_acc = motions_a
        motions_b_acc = motions_b

    # print processed motions
    print(f"Number of synchronous samples: {len(motions_a_acc)}")
    if len(motions_a_acc) < 2:
        print("\nerror: cannot calibrate with less than 2 samples")
        sys.exit(-1)

    # calibrate
    print("Calibrate")
    calib, metrics = calibrate(method, motions_a, motions_b, calib_gt)
    if calib is None:
        sys.exit(-1)

    # print result
    print("\n== Result ==")
    print(f"Frames:           {frame_a} -> {frame_b}")
    ev.print_calib(calib, norm=True)
    if 'scale' in metrics and metrics['scale'] is not None:
        if isinstance(metrics['scale'], list):
            scale_str = ' | '.join([f"{s:.2f}" for s in metrics['scale']])
        else:
            scale_str = f"{metrics['scale']:.2f}"
        print(f"Scale (b):        {scale_str}")

    print("\n== Metrics ==")
    ev.print_metrics(metrics)

    # export calibration
    if args.output is not None:
        print(f"\nExport calibration to '{args.output}'")
        manager = CalibrationManager()
        manager.add(frame_a, frame_b, calib)
        manager.save(args.output)

    # visualization
    if not args.novis:
        # rotation angles
        plot_rotation_angles(motions_a_acc, motions_b_acc, frame_a, frame_b)

        # sample errors
        if 'errors' in metrics and metrics['errors'] is not None:
            ev.plot_ransac_errors(metrics['errors'], method, log=True)

        plt.show()


if __name__ == '__main__':
    main()
