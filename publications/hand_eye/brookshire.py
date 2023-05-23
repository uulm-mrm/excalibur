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
BROOKSHIRE_DIR = osp.join(SCRIPT_DIR, os.pardir, 'data', 'brookshire')


def main():
    # load data
    transforms1, transforms2, calib_gt = io_proc.load_transform_pair(
        osp.join(BROOKSHIRE_DIR, 'sensor1.m3d'), osp.join(BROOKSHIRE_DIR, 'sensor2.m3d'))

    # processing
    pair_proc = Compose([
        transform_proc.PosesToMotions(),
        transform_proc.Normalized()
    ])
    motions1, motions2 = pair_proc((transforms1, transforms2))

    calib_gt.normalized_()

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


if __name__ == '__main__':
    main()
