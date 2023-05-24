#!/usr/bin/env python3
import os
import os.path as osp

import numpy as np
import pandas as pd

from excalibur.processing import Compose, io as io_proc, transforms as transform_proc
from excalibur.visualization.table import ColumnFormat, print_results_table

import utils
from utils import MethodConfig


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
EUROC_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'euroc')


def run(sensor1, sensor2, sequence, scale_a=False):
    # load data
    transforms1, transforms2, calib_gt = io_proc.load_transform_pair(
        osp.join(EUROC_DIR, sensor1, f'{sequence}.m3d'), osp.join(EUROC_DIR, sensor2, f'{sequence}.m3d'))

    # processing
    pair_proc = Compose([
        transform_proc.MatchPoses(max_diff_nsec=1e9),
        transform_proc.PosesToMotions(),
        transform_proc.Normalized(),
    ])
    motions1, motions2 = pair_proc((transforms1, transforms2))

    # swap for scaling sensor a
    if scale_a:
        motions1, motions2 = motions2, motions1
        calib_gt.inverse_()

    # calibrate
    methods = {
        'QCQPDQ Fast': MethodConfig('DualQuaternionQCQPScaled', fast=True, reduced=True),
        'QCQPDQ Glob': MethodConfig('DualQuaternionQCQPScaled', fast=False, reduced=True),
        'QCQPMat': MethodConfig('MatrixQCQPScaled'),
        'Wei': MethodConfig('Wei'),
        'SchmidtDQ': MethodConfig('SchmidtDQ', improved=False),
        'SchmidtDQ (Fast)': MethodConfig('SchmidtDQ', improved=True),
        'SchmidtHM': MethodConfig('SchmidtHM', improved=False),
        'SchmidtHM (Fast)': MethodConfig('SchmidtHM', improved=True),
    }

    results = [{'method': method, **utils.calibrate(cfg, motions1, motions2, calib_gt)}
               for method, cfg in methods.items()]

    # table
    df = pd.DataFrame(results)
    df.set_index('method', inplace=True)

    column_formats = {
        't_err': ColumnFormat('t_err [cm]', lambda x: f'{x * 1e2:.2f}'),
        'r_err': ColumnFormat('r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'scale': ColumnFormat('scale', lambda x: f'{x:.3f}'),
        'time': ColumnFormat('time [ms]', lambda x: f'{x * 1e3:.1f}'),
        'is_global': ColumnFormat('Global'),
    }
    print_results_table(df, column_formats)


def main():
    sequences = [
        'MH_01_easy',
        'MH_02_easy',
        'MH_03_medium',
    ]

    for seq in sequences:
        print(f"==== {seq} ====")
        run('cam0_mono_orbslam', 'ground_truth', seq, scale_a=True)


if __name__ == '__main__':
    main()
