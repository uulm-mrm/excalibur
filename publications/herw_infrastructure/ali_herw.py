#!/usr/bin/env python3
import argparse
import os
import os.path as osp

import motion3d as m3d
import numpy as np
import pandas as pd

import excalibur as excal
from excalibur.io.calibration import load_calibration
from excalibur.processing import io as io_proc
from excalibur.visualization.table import ColumnFormat, print_results_table

import utils
from utils import MethodConfig


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
HERW_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'herw_ali')


def _run_methods(poses_a, poses_b, calib_x, calib_y, init_method, add_noise=False, detections=None, intrinsics=None):
    if add_noise:
        poses_a_noisy = m3d.TransformContainer(has_stamps=False, has_poses=True)
        poses_b_noisy = m3d.TransformContainer(has_stamps=False, has_poses=True)
        for pa, pb in zip(poses_a, poses_b):
            # noise values
            translation_noise = np.random.normal(0.0, 0.01, (3, 1))
            rotation_noise_rad = np.deg2rad(np.random.normal(0.0, 0.1))
            rotation_noise_axis = np.random.normal(loc=0.0, scale=1.0, size=3)
            rotation_noise_axis /= np.linalg.norm(rotation_noise_axis)

            # noisy transform
            noise = m3d.AxisAngleTransform(translation_noise, rotation_noise_rad, rotation_noise_axis)
            poses_a_noisy.append(pa * noise)

            # target transform
            poses_b_noisy.append(pb)
        poses_a = poses_a_noisy
        poses_b = poses_b_noisy

    # initial value for methods with required init
    method_init = excal.calibration.HERWCalibrationBase.create(init_method)
    method_init.set_transforms(poses_a, poses_b)
    result_init = method_init.calibrate()

    # calibrate
    methods = {
        'DualQuaternionQCQP': MethodConfig('DualQuaternionQCQPSignSampling',
                                           init_kwargs={'n_iter': 20, 'use_C': True}),
        'LiDQ': MethodConfig('LiDQSignSampling', init_kwargs={'n_iter': 20}),
        'LiHM': MethodConfig('LiHM'),
        'Shah': MethodConfig('Shah'),
        'Wang': MethodConfig('Wang'),
        'Dornaika': MethodConfig('Dornaika', calib_kwargs={'calib_init': result_init.calib}),
        'Tabb 1': MethodConfig('Tabb', calib_kwargs={'calib_init': result_init.calib, 'use_cost2': False}),
        'Tabb 2': MethodConfig('Tabb', calib_kwargs={'calib_init': result_init.calib, 'use_cost2': True}),
    }

    results = [{'method': name, **utils.calibrate_herw(cfg, poses_a, poses_b, calib_x, calib_y,
                                                       detections=detections, intrinsics=intrinsics)}
               for name, cfg in methods.items()]
    return results


def run_multi(dataset, n_runs, init_method, add_noise=False, is_real=False):
    # input files
    a_filename = osp.join(HERW_DIR, dataset, 'a.m3d')
    b_filename = osp.join(HERW_DIR, dataset, 'b.m3d')
    calib_filename = osp.join(HERW_DIR, dataset, 'calib.yaml')
    detections_filename = osp.join(HERW_DIR, dataset, 'detections.npy')
    intrinsics_filename = osp.join(HERW_DIR, dataset, 'intrinsics.yaml')

    # load data
    poses_a, poses_b, _ = io_proc.load_transform_pair(
        a_filename, b_filename, normalized=True)

    # load detections
    if osp.exists(detections_filename):
        detections = np.load(detections_filename, allow_pickle=True).item()
    else:
        detections = None

    # load intrinsics
    if osp.exists(intrinsics_filename):
        intrinsics = excal.io.camera.CameraIntrinsics.load(intrinsics_filename)
    else:
        intrinsics = None

    # load ground truth
    calib_data = load_calibration(calib_filename)
    if calib_data is None:
        calib_x = None
        calib_y = None
    else:
        calib_x = calib_data['x']
        calib_y = calib_data['y']

    # run methods
    results = []
    for run_id in range(n_runs):
        print(f"\rRun {run_id + 1}/{n_runs}", end="", flush=True)
        results.extend(_run_methods(poses_a, poses_b, calib_x, calib_y, init_method, add_noise,
                                    detections, intrinsics))
    print("")

    # combine
    df = pd.DataFrame(results)
    df_grouped = df.groupby('method')
    df_lists = df_grouped.aggregate(list)

    # show
    if is_real:
        column_formats_pretty = {
            't_errs_cycle': ColumnFormat('C t_err [mm]',
                                        lambda x: f'{np.mean(x) * 1e3:.1f} ± {np.std(x) * 1e3:.1f}'),
            'r_errs_cycle': ColumnFormat('C r_err [deg]',
                                    lambda x: f'{np.rad2deg(np.mean(x)):.2f} ± {np.rad2deg(np.std(x)):.2f}'),
            'reprojection_error': ColumnFormat('Rep [px]',
                                    lambda x: f'{np.mean(x):.2f} ± {np.std(x):.2f}'),
            'time': ColumnFormat('time [ms]',
                                 lambda x: f'{np.mean(x) * 1e3:.1f} ± {np.std(x) * 1e3:.1f}'),
            'gap': ColumnFormat('Max. Gap', lambda x: f'{np.max(np.abs(x)):.2E}'),
        }
    else:
        column_formats_pretty = {
            't_err_x': ColumnFormat('X t_err [mm]',
                                    lambda x: f'{np.mean(x) * 1e3:.1f} ± {np.std(x) * 1e3:.1f}'),
            'r_err_x': ColumnFormat('X r_err [deg]',
                                    lambda x: f'{np.rad2deg(np.mean(x)):.2f} ± {np.rad2deg(np.std(x)):.2f}'),
            't_err_y': ColumnFormat('Y t_err [mm]',
                                    lambda x: f'{np.mean(x) * 1e3:.1f} ± {np.std(x) * 1e3:.1f}'),
            'r_err_y': ColumnFormat('Y r_err [deg]',
                                    lambda x: f'{np.rad2deg(np.mean(x)):.2f} ± {np.rad2deg(np.std(x)):.2f}'),
            'time': ColumnFormat('time [ms]',
                                 lambda x: f'{np.mean(x) * 1e3:.1f} ± {np.std(x) * 1e3:.1f}'),
            'gap': ColumnFormat('Max. Gap', lambda x: f'{np.max(np.abs(x)):.2E}'),
        }
    print_results_table(df_lists, column_formats_pretty)


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nruns', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    n_runs = args.nruns
    if args.seed is not None:
        np.random.seed(args.seed)

    # datasets
    datasets = {
        'CS_synthetic_1': False,
        'kuka_2': True
    }

    # calibrate
    for dataset, is_real in datasets.items():
        print(f"\n==== {dataset} without noise and {n_runs} runs ====")
        run_multi(dataset, n_runs, init_method='LiHM', add_noise=False, is_real=is_real)

    print(f"\n==== CS_synthetic_1 with noise and {n_runs} runs ====")
    run_multi('CS_synthetic_1', n_runs, init_method='LiHM', add_noise=True)


if __name__ == '__main__':
    main()
