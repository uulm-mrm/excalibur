#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import itertools
import os.path as osp
from pathlib import Path

import motion3d as m3d
import numpy as np
import pandas as pd

import excalibur as excal
from excalibur.calibration.herw.base import HERWData
from excalibur.io.calibration import load_calibration
from excalibur.io.geometry import load_line_data
from excalibur.processing import io as io_proc
from excalibur.visualization.table import ColumnFormat, print_results_table

import utils
from utils import MethodConfig


SCRIPT_PATH = Path(osp.dirname(osp.realpath(__file__)))
LEHR_PATH = SCRIPT_PATH.parent / 'data' / 'lehr'

LEHR_ORIGIN_POS = np.array([571659.0435911490, 5364823.651433706, 595.9])
LEHR_ORIGIN_INV = m3d.EulerTransform(-LEHR_ORIGIN_POS, 0, 0, 0, m3d.EulerAxes.kSXYZ)


def _find_vehicle_for_target(vehicles, target):
    for vehicle, targets in vehicles.items():
        if target in targets:
            return vehicle
    return None


def _apply_blacklist(transforms, blacklist, threshold=1000):
    transforms_filtered = m3d.TransformContainer(has_stamps=True, has_poses=True)
    for stamp, transform, in transforms.items():
        stamp_ns = stamp.toNSec()
        blacklist_found = False
        for bl_entry in blacklist:
            if isinstance(bl_entry, tuple):
                if bl_entry[0] - threshold <= stamp_ns <= bl_entry[1] + threshold:
                    blacklist_found = True
                    continue
            else:
                if np.abs(stamp_ns - bl_entry) < threshold:
                    blacklist_found = True
                    continue
        if blacklist_found:
            continue
        transforms_filtered.insert(stamp, transform)
    return transforms_filtered


@dataclass
class TargetDetections:
    sensor: str
    target: str
    transforms: m3d.TransformContainer


def run_single(targets, sensors, methods, vehicles, blacklists, translation_norms=None):
    # load data
    vehicle_transforms = {
        veh: {sensor: io_proc.load_transforms_file(str(LEHR_PATH / sensor / f'{veh}.m3d'), normalized=True).applyPre(LEHR_ORIGIN_INV)
              for sensor in sensors}
        for veh in vehicles.keys()
    }

    target_detections = [
        TargetDetections(
            sensor=sensor, target=target,
            transforms=io_proc.load_transforms_file(str(LEHR_PATH / sensor / f'{target}.m3d'), normalized=True))
        for sensor, target in itertools.product(sensors, targets)
        if (LEHR_PATH / sensor / f'{target}.m3d').is_file()
    ]

    intrinsics = {
        sensor: excal.io.camera.CameraIntrinsics.load(LEHR_PATH / sensor / 'intrinsics.yaml')
        for sensor in sensors
        if (LEHR_PATH / sensor / 'intrinsics.yaml').is_file()
    }

    lines = {
        sensor: load_line_data(LEHR_PATH / sensor / 'lines.txt')
        for sensor in sensors
        if (LEHR_PATH / sensor / 'lines.txt').is_file()
    }

    calib_data = load_calibration(LEHR_PATH / 'calib.yaml')

    # filter data
    vehicle_transforms = {
        veh: {sensor: _apply_blacklist(transforms, blacklist=blacklists[veh])
              for sensor, transforms in veh_data.items()}
        for veh, veh_data in vehicle_transforms.items()
    }

    # process data
    target_vehicles = {target: _find_vehicle_for_target(vehicles, target) for target in targets}

    match_poses = excal.processing.transforms.MatchPoses()

    def _flip(a, b):
        return {'transforms_a': b, 'transforms_b': a}

    transform_data = [
        HERWData(
            frame_x=det.target, frame_y=det.sensor,
            **_flip(*match_poses((det.transforms, vehicle_transforms[target_vehicles[det.target]][det.sensor]))))
        for det in target_detections
    ]

    ground_truth_x = {target: calib_data[target]
                      for target in targets if target in calib_data}
    ground_truth_y = {sensor: calib_data[sensor].applyPre(LEHR_ORIGIN_INV)
                      for sensor in sensors if sensor in calib_data}

    # extract target translation norms
    if translation_norms is None:
        translation_norms = {
            name: transform.translationNorm()
            for name, transform in ground_truth_x.items()
        }

    # methods for single sensor-target pair
    methods_single = {
        'Dornaika': MethodConfig('Dornaika',),
        'LiDQ': MethodConfig('LiDQSignSampling',),
        'LiHM': MethodConfig('LiHM'),
        'Tabb': MethodConfig('Tabb'),
        'Wang': MethodConfig('Wang'),
        'F2F': MethodConfig('F2F'),
        'PnP': MethodConfig('PnP'),
    }

    results = []
    for name, cfg in methods_single.items():
        if name not in methods:
            continue
        assert len(transform_data) == 1
        assert len(lines) == 1
        assert len(intrinsics) == 1

        # calibrate
        method_results = utils.calibrate_herw(
            cfg,
            transform_data[0].transforms_a,
            transform_data[0].transforms_b,
            ground_truth_x[transform_data[0].frame_x],
            None,
            lines=lines[transform_data[0].frame_y],
            intrinsics=intrinsics[transform_data[0].frame_y],
        )

        # attach frame ids
        def attach_frame_id(data, main_key, frame_id):
            data[f'{main_key}_{frame_id}'] = data[main_key]
            del data[main_key]
        attach_frame_id(method_results, 't_err_x', transform_data[0].frame_x)
        attach_frame_id(method_results, 'r_err_x', transform_data[0].frame_x)
        attach_frame_id(method_results, 't_err_y', transform_data[0].frame_y)
        attach_frame_id(method_results, 'r_err_y', transform_data[0].frame_y)
        attach_frame_id(method_results, 'rel_line_error', transform_data[0].frame_y)
        attach_frame_id(method_results, 'reprojection_error', transform_data[0].frame_y)

        # store
        results.append({'method': name, **method_results})

    # methods for multiple targets / sensors
    calib_args_dq_qcqp = {'dual_rec_kwargs': {'eps_constraints': 1e-3}}

    methods_multi = {
        'Ours (w/o X norm)': MethodConfig('DualQuaternionQCQPSignSampling',
                                          init_kwargs={'n_iter': 20, 'use_C': True},
                                          calib_kwargs={**calib_args_dq_qcqp}, force_positive_z=False),
        'Ours': MethodConfig('DualQuaternionQCQPSignSampling',
                             init_kwargs={'n_iter': 20, 'use_C': True},
                             calib_kwargs={'t_norms': translation_norms, **calib_args_dq_qcqp},
                             force_positive_z=True),
    }

    for name, cfg in methods_multi.items():
        if name not in methods:
            continue

        # calibrate
        method_results = utils.calibrate_herw_multi(
            cfg,
            transform_data,
            ground_truth_x,
            ground_truth_y,
            lines=lines,
            intrinsics=intrinsics,
        )

        # attach frame ids
        def flatten_dict(data, main_key, new_main=None):
            if new_main is None:
                new_main = main_key
            for k, v in data[main_key].items():
                data[f'{new_main}_{k}'] = v
            del data[main_key]
        flatten_dict(method_results, 't_errs_x', 't_err_x')
        flatten_dict(method_results, 'r_errs_x', 'r_err_x')
        flatten_dict(method_results, 't_errs_y', 't_err_y')
        flatten_dict(method_results, 'r_errs_y', 'r_err_y')
        flatten_dict(method_results, 'rel_line_errors', 'rel_line_error')
        flatten_dict(method_results, 'reprojection_errors', 'reprojection_error')

        # store
        results.append({'method': name, **method_results})

    return results


def run_multi(n_runs, *args, **kwargs):
    # run multiple times
    single_results = []
    for run_id in range(n_runs):
        print(f"Run {run_id + 1}/{n_runs}")
        single_results.extend(run_single(*args, **kwargs))
    print("")

    # aggregate
    df = pd.DataFrame(single_results)
    df_grouped = df.groupby('method')
    df_agg = df_grouped.aggregate(lambda x: np.mean(np.array(x)))
    return df_agg


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nruns', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    n_runs = args.nruns
    if args.seed is not None:
        np.random.seed(args.seed)

    # vehicles
    vehicles = {
        'vehicle': ['chessboard', 'aruco0', 'aruco1']
    }
    blacklists = {
        'vehicle': [
            (1662569750700000256, 1662569800600000256),
        ]
    }

    # translation
    translation_norms_est = {
        'chessboard': np.linalg.norm([1.02, 0.0, 1.58])
    }

    # runs
    runs = [
        {'targets': ['chessboard'], 'sensors': ['spu2_cam1'],
         'methods': ['Ours', 'Ours (w/o X norm)', 'Dornaika', 'LiDQ', 'LiHM', 'Tabb', 'Wang', 'F2F', 'PnP']},
        {'targets': ['chessboard'], 'sensors': ['spu2_cam2'], 'methods': ['Ours', 'F2F', 'PnP']},
        {'targets': ['chessboard'], 'sensors': ['spu2_cam1', 'spu2_cam2'], 'methods': ['Ours']},
        {'targets': ['chessboard', 'aruco0'], 'sensors': ['spu2_cam1', 'spu2_cam2'], 'methods': ['Ours']},
    ]

    # execute
    run_df_list = []
    for run_id, run_data in enumerate(runs):
        # run
        print(f"## Targets: {', '.join(run_data['targets'])}  |  "
              f"Sensors: {', '.join(run_data['sensors'])}  |  "
              f"Methods: {', '.join(run_data['methods'])}  |  ")
        run_df = run_multi(n_runs, run_data['targets'], run_data['sensors'], run_data['methods'], vehicles,
                           blacklists, translation_norms_est)

        # attach run id, target, and sensor names to results
        run_df['run_id'] = run_id
        run_df['targets'] = ' | '.join(run_data['targets'])
        run_df['sensors'] = ' | '.join(run_data['sensors'])

        # append to results
        run_df_list.append(run_df)

    # concatenate
    df = pd.concat(run_df_list)

    # show
    column_formats = {
        'run_id': ColumnFormat('Run', lambda x: f'{x}'),
        'method': ColumnFormat('Method', lambda x: f'{x}'),
        'targets': ColumnFormat('Targets', lambda x: f'{x}'),
        'sensors': ColumnFormat('Sensors', lambda x: f'{x}'),
        't_err_x_chessboard': ColumnFormat('X t_err [cm]', lambda x: f'{x * 1e2:.1f}'),
        'r_err_x_chessboard': ColumnFormat('X r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'rel_line_error_spu2_cam1': ColumnFormat('Line RMSE Cam1 [%]', lambda x: f'{x * 1e2:.1f}'),
        'reprojection_error_spu2_cam1': ColumnFormat('Reproj Err Cam1 [px]', lambda x: f'{x:.2f}'),
        'rel_line_error_spu2_cam2': ColumnFormat('Line RMSE Cam2 [%]', lambda x: f'{x * 1e2:.1f}'),
        'reprojection_error_spu2_cam2': ColumnFormat('Reproj Err Cam2 [px]', lambda x: f'{x:.2f}'),
        'time': ColumnFormat('Time [ms]', lambda x: f'{x * 1e3:.1f}'),
        'gap': ColumnFormat('Gap', lambda x: f'{x:.2E}'),
    }
    print_results_table(df, column_formats)


if __name__ == '__main__':
    main()
