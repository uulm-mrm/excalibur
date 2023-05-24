#!/usr/bin/env python3
import os
import os.path as osp

import numpy as np
import pandas as pd

from excalibur.processing import ApplyToList, Compose, Lambda, io as io_proc, transforms as transform_proc
from excalibur.visualization.table import ColumnFormat, print_results_table

import utils
from utils import MethodConfig


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
EUROC_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'euroc')


def merge_list_of_tuples(data):
    assert isinstance(data, list)
    output = None

    for sample in data:
        assert isinstance(sample, tuple)
        if output is None:
            output = [[s] for s in sample]
        else:
            assert len(output) == len(sample)
            for i, s in enumerate(sample):
                output[i].append(s)

    return tuple(output)


def run(sensor1, sensor2, sequences, scale_a=False):
    # input files
    sensor1_dir = osp.join(EUROC_DIR, sensor1)
    sensor2_dir = osp.join(EUROC_DIR, sensor2)
    filenames = [(osp.join(sensor1_dir, f'{seq}.m3d'), osp.join(sensor2_dir, f'{seq}.m3d'))
                 for seq in sequences]

    # load data
    transforms, calib_gt = io_proc.load_transforms(filenames)

    # processing
    pair_proc = Compose([
        transform_proc.MatchPoses(max_diff_nsec=1e9),
        transform_proc.PosesToMotions(),
        transform_proc.Normalized(),
    ])

    proc = Compose([
        ApplyToList(pair_proc),
        Lambda(merge_list_of_tuples),
    ])

    motions_list1, motions_list2 = proc(transforms)

    # swap for scaling sensor a
    if scale_a:
        motions_list1, motions_list2 = motions_list2, motions_list1
        calib_gt.inverse_()

    # calibrate
    methods = {
        'QCQPDQ Fast': MethodConfig('DualQuaternionQCQPScaled', fast=True, reduced=True),
        'QCQPDQ Glob': MethodConfig('DualQuaternionQCQPScaled', fast=False, reduced=True),
    }

    results = [{'method': method, **utils.calibrate(cfg, motions_list1, motions_list2, calib_gt)}
               for method, cfg in methods.items()]

    # table
    df = pd.DataFrame(results)
    df.set_index('method', inplace=True)

    column_formats = {
        't_err': ColumnFormat('t_err [cm]', lambda x: f'{x * 1e2:.2f}'),
        'r_err': ColumnFormat('r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'scale': ColumnFormat('scales', lambda x_list: ' | '.join([f'{x:.3f}' for x in x_list])),
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

    run('cam0_mono_orbslam', 'ground_truth', sequences, scale_a=True)


if __name__ == '__main__':
    main()
