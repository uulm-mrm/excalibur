#!/usr/bin/env python3
import os
import os.path as osp

import numpy as np
import pandas as pd

from excalibur.fitting.plane import Plane
from excalibur.processing import ApplyToList, Compose, io as io_proc, transforms as transform_proc
from excalibur.visualization.table import ColumnFormat, merge_to_multiindex_column, print_results_table

import utils
from utils import MethodConfig


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
KITTI_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'kitti_odometry')


def run(sensor1, sensor2, plane1_name, plane2_name, sequences):
    # input files
    sensor1_dir = osp.join(KITTI_DIR, sensor1)
    sensor2_dir = osp.join(KITTI_DIR, sensor2)
    plane1_file = osp.join(KITTI_DIR, 'ground_planes', f'{plane1_name}.json')
    plane2_file = osp.join(KITTI_DIR, 'ground_planes', f'{plane2_name}.json')
    filenames = [(osp.join(sensor1_dir, f'{seq:02}.m3d'), osp.join(sensor2_dir, f'{seq:02}.m3d'))
                 for seq in sequences]

    # load data
    transforms, calib_gt = io_proc.load_transforms(filenames)
    plane1 = Plane.load(plane1_file)
    plane2 = Plane.load(plane2_file)

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
        'QCQPDQ Glob w/ Norm': MethodConfig(
            'DualQuaternionQCQP',
            t_norm=calib_gt.translationNorm(),
            up_approx=plane1.normal,
        ),
    }
    results_nonplanar = [{'method': name, **utils.calibrate(cfg, motions1, motions2, calib_gt)}
                         for name, cfg in methods.items()]

    planar_methods = {
        'QCQPDQ Glob': MethodConfig('DualQuaternionQCQPPlanar', fast=False),
        'QCQPDQ Fast': MethodConfig('DualQuaternionQCQPPlanar', fast=True),
    }
    results_planar = [{'method': name, **utils.calibrate_planar(cfg, motions1, motions2,
                                                                plane1.get_transform(), plane2.get_transform(),
                                                                calib_gt)}
                      for name, cfg in planar_methods.items()]

    # merge results
    df_nonplanar = pd.DataFrame(results_nonplanar)
    df_planar = pd.DataFrame(results_planar)
    df = merge_to_multiindex_column({'non-planar': df_nonplanar, 'planar': df_planar}, index='method')

    # show
    column_formats = {
        't_err': ColumnFormat('t_err [cm]', lambda x: f'{x * 1e2:.2f}'),
        'r_err': ColumnFormat('r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'time': ColumnFormat('time [ms]', lambda x: f'{x * 1e3:.1f}'),
        'is_global': ColumnFormat('Global'),
        'trans_cond': ColumnFormat('Cond', lambda x: f'{x:.1f}'),
    }
    print_results_table(df, column_formats)


def main():
    run('cam0_stereo_orbslam', 'velodyne_aloam', '04-11_cam0', '04-11_velodyne', [10])


if __name__ == '__main__':
    main()
