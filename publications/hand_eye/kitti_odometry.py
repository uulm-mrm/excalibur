#!/usr/bin/env python3
import os
import os.path as osp

import numpy as np
import pandas as pd

from excalibur.processing import ApplyToList, Compose, io as io_proc, transforms as transform_proc
from excalibur.visualization.table import ColumnFormat, print_results_table

import utils
from utils import MethodConfig


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
KITTI_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'kitti_odometry')


def run(sensor1, sensor2, sequences):
    # input files
    sensor1_dir = osp.join(KITTI_DIR, sensor1)
    sensor2_dir = osp.join(KITTI_DIR, sensor2)
    filenames = [(osp.join(sensor1_dir, f'{seq:02}.m3d'), osp.join(sensor2_dir, f'{seq:02}.m3d'))
                 for seq in sequences]

    # load data
    transforms, calib_gt = io_proc.load_transforms(filenames)

    # processing
    pair_proc = Compose([
        transform_proc.MatchPoses(max_diff_nsec=0.1 * 1e9),
        transform_proc.PosesToMotions(max_step_sec=1.0),
        transform_proc.Normalized(),
    ])

    proc = Compose([
        ApplyToList(pair_proc),
        transform_proc.MergeTransforms()
    ])

    motions1, motions2 = proc(transforms)

    # calibrate
    methods = {
        'QCQPDQ Glob': MethodConfig('DualQuaternionQCQP', fast=False),
        'QCQPDQ Fast': MethodConfig('DualQuaternionQCQP', fast=True),
        'Daniilidis': MethodConfig('Daniilidis'),
        'Andreff': MethodConfig('Andreff'),
        'MatrixQCQP': MethodConfig('MatrixQCQP'),
    }
    results = [{'method': method, **utils.calibrate(cfg, motions1, motions2, calib_gt)}
               for method, cfg in methods.items()]

    # table
    df = pd.DataFrame(results)
    df.set_index('method', inplace=True)

    column_formats = {
        't_err': ColumnFormat('t_err [cm]', lambda x: f'{x * 1e2:.2f}'),
        'r_err': ColumnFormat('r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'time': ColumnFormat('time [ms]', lambda x: f'{x * 1e3:.1f}'),
        'is_global': ColumnFormat('Global'),
    }
    print_results_table(df, column_formats)


def main():
    runs = {
        '07': [7],
    }
    sensor_pairs = [
        ('ground_truth', 'velodyne_aloam'),
        ('cam0_stereo_openvslam', 'ground_truth'),
        # ('cam0_stereo_orbslam', 'ground_truth'),
        ('cam0_stereo_openvslam', 'velodyne_aloam'),
        # ('cam0_stereo_orbslam', 'velodyne_aloam'),
    ]

    for run_name, run_sequences in runs.items():
        print(f"==== {run_name} ====")
        for sensor_pair in sensor_pairs:
            print(f"-- {sensor_pair[0]} -> {sensor_pair[1]} --")
            run(*sensor_pair, run_sequences)


if __name__ == '__main__':
    main()
