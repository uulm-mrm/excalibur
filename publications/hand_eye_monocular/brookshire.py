#!/usr/bin/env python3
import itertools
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


def run(sensor_a, sensor_b):
    # load data
    transforms1, transforms2, calib_gt = io_proc.load_transform_pair(
        osp.join(BROOKSHIRE_DIR, f'{sensor_a}.m3d'), osp.join(BROOKSHIRE_DIR, f'{sensor_b}.m3d'))

    # processing
    pair_proc = Compose([
        transform_proc.PosesToMotions(max_step_sec=None),
        transform_proc.Normalized()
    ])
    motions1, motions2 = pair_proc((transforms1, transforms2))

    calib_gt.normalized_()

    # calibrate
    scalings = [1.0, 10.0, 0.01]

    qcqp_dq_args = {'reduced': False, 'dual_rec_kwargs': {'eps_constraints': 1e-3}}
    qcqp_dq_fast_args = {'qcqp_kwargs': {'solver_kwargs': {'tol': 1e-7}}}
    schmidt_hm_args = {'solver_kwargs': {'options': {'maxiter': 1000}}}
    methods = {
        'Horn': MethodConfig('DualQuaternionQCQP'),
        'QCQPDQ Fast': MethodConfig('DualQuaternionQCQPScaled', fast=True, **qcqp_dq_args, **qcqp_dq_fast_args),
        'QCQPDQ Glob': MethodConfig('DualQuaternionQCQPScaled', fast=False, **qcqp_dq_args),
        'QCQPMat': MethodConfig('MatrixQCQPScaled'),
        'Wei': MethodConfig('Wei'),
        'SchmidtDQ': MethodConfig('SchmidtDQ', improved=False),
        'SchmidtDQ (Fast)': MethodConfig('SchmidtDQ', improved=True),
        'SchmidtHM': MethodConfig('SchmidtHM', improved=False, **schmidt_hm_args),
        'SchmidtHM (Fast)': MethodConfig('SchmidtHM', improved=True, **schmidt_hm_args),
    }

    results = [
        {'method': method, 'scale_gt': str(scale),
         **utils.calibrate(cfg, motions1, motions2.scaleTranslation(1.0 / scale), calib_gt)}
        for (method, cfg), scale in itertools.product(methods.items(), scalings)
    ]

    # table
    df = pd.DataFrame(results)
    df = df.pivot(index='method', columns='scale_gt')
    df = df.swaplevel(0, 1, 1).sort_index(axis=1)

    # show
    column_formats = {
        't_err': ColumnFormat('t_err [cm]', lambda x: f'{x * 1e2:.2f}'),
        'r_err': ColumnFormat('r_err [deg]', lambda x: f'{np.rad2deg(x):.2f}'),
        'scale': ColumnFormat('scale', lambda x: f'{x:.3f}'),
        'time': ColumnFormat('time [ms]', lambda x: f'{x * 1e3:.1f}'),
    }
    print_results_table(df, column_formats)


def main():
    print("==== Sensor1 -> Sensor2 ====")
    run('sensor1', 'sensor2')

    print("==== Sensor2 -> Sensor1 ====")
    run('sensor2', 'sensor1')


if __name__ == '__main__':
    main()
