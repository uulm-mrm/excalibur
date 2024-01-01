from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np
try:
    import tikzplotlib
except ImportError:
    import warnings
    warnings.warn("Could not import tikzplotlib.")
import yaml

from excalibur.calibration.base import _CalibrationBase
from excalibur.calibration.utils.ransac import RANSACMethod
from excalibur.optimization.qcqp import QCQPSolveMethod
from excalibur.visualization.table import ColumnFormat, print_results_table


@dataclass
class MethodConfig:
    name: str
    init_kwargs: Optional[Dict] = field(default_factory=dict)
    calib_kwargs: Optional[Dict] = field(default_factory=dict)
    extra_fun: Optional[Callable] = None
    extra_kwargs: Optional[Dict] = field(default_factory=dict)

    @classmethod
    def load(cls, filename: str):
        # load yaml
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        # initialize config
        if 'name' not in data:
            raise RuntimeError("Missing 'name' in config.")
        config = cls(name=data['name'])
        del data['name']

        # set kwargs
        if 'init_kwargs' in data:
            config.init_kwargs = data['init_kwargs']
            del data['init_kwargs']
        if 'calib_kwargs' in data:
            config.calib_kwargs = data['calib_kwargs']
            del data['calib_kwargs']

        # ransac
        if 'ransac' in data:
            ransac_data = data['ransac']

            # base
            if 'base' not in ransac_data:
                raise RuntimeError("Missing 'base' in RANSAC config.")
            base_ransac = RANSACMethod.from_str(ransac_data['base'])
            del ransac_data['base']

            # config
            config = get_ransac_method_config(base_ransac, config, **ransac_data)
            del data['ransac']

        # extra arguments
        config.extra_kwargs = data
        return config

    def print(self):
        print(f"Name: {self.name}")
        if self.init_kwargs:
            print("\nInitialiation")
            print(yaml.dump(self.init_kwargs, default_flow_style=False))
        if self.calib_kwargs:
            print("\nCalibration")
            print(yaml.dump(self.calib_kwargs, default_flow_style=False))
        if self.extra_kwargs:
            print("\nExtra")
            print(yaml.dump(self.extra_kwargs, default_flow_style=False))

    def create(self, base_cls: _CalibrationBase) -> _CalibrationBase:
        method = base_cls.create(self.name, **self.init_kwargs)
        method.configure(**self.calib_kwargs)
        if self.extra_fun is not None:
            self.extra_fun(method)
        return method

    def get_extra_kwarg(self, key, required=False, default=None):
        if key not in self.extra_kwargs:
            if required:
                raise KeyError(f"Required key '{key}' not found in extra arguments.")
            else:
                return default
        return self.extra_kwargs[key]


def get_ransac_method_config(base_ransac: RANSACMethod, method_cfg: MethodConfig, trans_thresh: Optional[float] = None,
                             rot_thresh: Optional[float] = None, rot_thresh_deg: Optional[float] = None,
                             nreps: int = 20, seed: Optional[int] = None, **kwargs) -> MethodConfig:
    # rotation threshold from deg
    if rot_thresh is None and rot_thresh_deg is not None:
        rot_thresh = float(np.deg2rad(rot_thresh_deg))

    # ransac config
    return MethodConfig(
        str(base_ransac.value),
        init_kwargs={
            'method_name': method_cfg.name,
            **method_cfg.init_kwargs,
            'nreps': nreps,
            'trans_thresh': trans_thresh,
            'rot_thresh': rot_thresh,
            'seed': seed,
            **kwargs,
        },
        calib_kwargs=method_cfg.calib_kwargs
    )


def _flatten(x):
    if isinstance(x[0], list):
        return [item for sublist in x for item in sublist if sublist is not None]
    else:
        return x


def mean_fmt(x, fmt, ignore_nan=False):
    if isinstance(x, list) and any(v is None for v in x):
        return '-'
    x = _flatten(x)
    mean = np.nanmean(x) if ignore_nan else np.mean(x)
    if np.isnan(mean):
        return '-'
    return f'{fmt(mean)}'


def mean_std_fmt(x, fmt, sym='±', ignore_nan=False):
    if isinstance(x, list) and any(v is None for v in x):
        return '-'
    x = _flatten(x)
    mean = np.nanmean(x) if ignore_nan else np.mean(x)
    std = np.nanstd(x) if ignore_nan else np.std(x)
    if np.isnan(std):
        if np.isnan(mean):
            return '-'
        return f'{fmt(mean)} {sym} n/a'
    return f'{fmt(mean)} {sym} {fmt(std)}'


def mean_std_tex_fmt(x, fun, num_dec):
    mean = np.mean(x)
    std = np.std(x)
    if np.isnan(std):
        if np.isnan(mean):
            return '-'
        return f'{round(fun(mean), num_dec)}'
    return f'{round(fun(mean), num_dec)} ({int(round(fun(std), num_dec) * (10 ** num_dec))})'


def gap_fmt(x):
    x = np.array(x)
    min_gap = np.min(x)
    max_gap = np.max(x)
    opt_count = np.sum(x <= 1e-4)
    opt_ratio = opt_count / len(x)
    return f'{min_gap:.2E} / {max_gap:.2E} | {opt_ratio * 100:.1f}% < 1e-4' if not np.isnan(max_gap) else '-'


def qcqpdq_method_fmt(x):
    x = [v.name if isinstance(v, QCQPSolveMethod) else '-'
         for v in x]
    names, counts = np.unique(x, return_counts=True)
    merged = [f'{n}({c * 100/ len(x):.1f}%)' for n, c in zip(names, counts)]
    return ','.join(merged)


def replace_method_names(df, name_dict):
    for old_name, new_name in name_dict.items():
        df['method'].replace(old_name, new_name, inplace=True)


def to_multi_column(df, columns):
    df_grouped = df.groupby(['method', *columns]).agg(list)
    df_grouped.reset_index(inplace=True)
    df_grouped.sort_values(by=['method'], inplace=True)

    df_grouped = df_grouped.pivot(index='method', columns=columns)
    df_grouped = df_grouped.swaplevel(0, 1, axis=1)

    if 'time' in df:
        df_time = df[['method', 'time']].groupby(['method']).agg(list)
        df_grouped[('', 'time_all')] = df_time

    return df_grouped


def print_check(df):
    # group to list
    df_grouped = df[['method', 'gap', 'qcqpdq_method']].groupby('method')
    df_gap = df_grouped.agg(list)

    # add count
    df_count = df_grouped.agg(len)
    df_gap['count'] = df_count['gap']

    # print
    column_formats = {
        'method': ColumnFormat('Method'),
        'gap': ColumnFormat('Gap', lambda x: gap_fmt(x)),
        'qcqpdq_method': ColumnFormat('QCQP', lambda x: qcqpdq_method_fmt(x)),
        'count': ColumnFormat('Count'),
    }
    print_results_table(df_gap, column_formats)


def _get_optimization_metrics(result):
    # initialize metric
    metrics = {'time': result.run_time}

    # specific metrics
    if 'qcqp_result' in result.aux_data:
        metrics['cost'] = result.aux_data['qcqp_result'].value
        metrics['gap'] = result.aux_data['qcqp_result'].gap
        metrics['qcqpdq_method'] = result.aux_data['qcqp_result'].method
    else:
        if 'cost' in result.aux_data:
            metrics['cost'] = result.aux_data['cost']
        if 'gap' in result.aux_data:
            metrics['gap'] = result.aux_data['gap']
    metrics['conditioning'] = result.aux_data['conditioning'].trans_cond if 'conditioning' in result.aux_data else None

    if 'cost' in metrics and 'gap' in metrics:
        metrics['gap_rel'] = metrics['gap'] / metrics['cost']

    # ransac
    metrics['ransac'] = True if 'inliers' in result.aux_data else False
    metrics['n_inliers'] = len(result.aux_data['inliers']) if 'inliers' in result.aux_data else None
    metrics['best_rep'] = result.aux_data['best_rep'] if 'best_rep' in result.aux_data else None
    metrics['errors'] = result.aux_data['errors'] if 'errors' in result.aux_data else None

    return metrics


def get_metrics_single(result, calib_x=None):
    # check success
    if not result.success:
        return None

    # error
    metrics = {}
    if calib_x is None:
        metrics['t_err'] = None
        metrics['r_err'] = None
    else:
        calib_err = result.calib.inverse() * calib_x
        metrics['t_err'] = calib_err.translationNorm()
        metrics['r_err'] = calib_err.rotationNorm()

    # optimization metrics
    metrics.update(_get_optimization_metrics(result))

    return metrics


def get_metrics_scale(result, calib_x=None, scaling_factor=None, scale_pred=None):
    # base metrics
    metrics = get_metrics_single(result, calib_x)
    if metrics is None:
        return None

    # additional
    if scale_pred is None:
        if hasattr(result, 'scale'):
            scale_pred = result.scale
        else:
            scale_pred = None

    metrics['scale_pred'] = scale_pred
    if scaling_factor is None:
        metrics['scale_err'] = None
    else:
        metrics['scale_err'] = np.abs((scale_pred - scaling_factor) / scaling_factor)\
            if scale_pred is not None else None

    return metrics


def get_metrics_multi(result, calib_x=None, calib_y=None, calib_x_pred=None, calib_y_pred=None,
                      x_frame=None, y_frame=None):
    # check success
    if not result.success:
        return None

    # predictions
    if calib_x_pred is None:
        if x_frame is None:
            calib_x_pred = result.calib.x
        else:
            calib_x_pred = result.calib.x[x_frame]
    if calib_y_pred is None:
        if y_frame is None:
            calib_y_pred = result.calib.y
        else:
            calib_y_pred = result.calib.y[y_frame]

    # errors
    metrics = {}
    if calib_x is not None:
        calib_err_x = calib_x_pred.inverse() * calib_x
        metrics['t_err_x'] = calib_err_x.translationNorm()
        metrics['r_err_x'] = calib_err_x.rotationNorm()
    if calib_y is not None:
        calib_err_y = calib_y_pred.inverse() * calib_y
        metrics['t_err_y'] = calib_err_y.translationNorm()
        metrics['r_err_y'] = calib_err_y.rotationNorm()

    # optimization metrics
    metrics.update(_get_optimization_metrics(result))

    return metrics


def _tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        _tikzplotlib_fix_ncols(child)


def export_tikz(filename, fig):
    _tikzplotlib_fix_ncols(fig)
    tikz_code = tikzplotlib.get_tikz_code(fig)
    print(f"Write figure to {filename}")
    with open(filename, 'w') as file:
        file.write(tikz_code)


def combine_filenames(filename_a=None, filename_b=None, list_a=None, list_b=None):
    filenames = []
    if filename_a is not None and filename_b is not None:
        filenames = [(filename_a, filename_b)]
    if list_a is not None and list_b is not None:
        if len(list_a) != len(list_b) != 0:
            print("error: the number of files for a and b must match.")
            return None
        filenames = [*filenames, *[(a, b) for a, b in zip(list_a, list_b)]]
    return filenames


def get_frames(filename_a, filename_b, file_frames, user_frames):
    # user input
    frame_a = user_frames[0]
    frame_b = user_frames[1]

    # file input
    if frame_a is None:
        frame_a = file_frames[0]
    if frame_b is None:
        frame_b = file_frames[1]

    # filenames
    if frame_a is None or frame_a == '':
        frame_a = Path(filename_a).stem
    if frame_b is None or frame_b == '':
        frame_b = Path(filename_b).stem

    # prevent duplicates
    if frame_a == frame_b:
        frame_a = f'{frame_a}_a'
        frame_b = f'{frame_b}_b'
    return frame_a, frame_b


def print_calib(transform, norm=False):
    euler = transform.asType(m3d.TransformType.kEuler)
    t_vec = euler.getTranslation()
    t_norm = np.linalg.norm(t_vec)
    angles = euler.getAngles()

    if t_norm >= 10.0:
        print(f"Translation [m]:  [{t_vec[0]:.2f}, {t_vec[1]:.2f}, {t_vec[2]:.2f}]")
        print(f"Angles [deg]:     "
              f"[{np.rad2deg(angles[0]):.1f}, {np.rad2deg(angles[1]):.1f}, {np.rad2deg(angles[2]):.1f}]")
        if norm:
            print(f"Trans. Norm [m]:  {np.linalg.norm(t_vec):.3f}")
    else:
        print(f"Translation [cm]: [{t_vec[0] * 100:.1f}, {t_vec[1] * 100:.1f}, {t_vec[2] * 100:.1f}]")
        print(f"Angles [deg]:     "
              f"[{np.rad2deg(angles[0]):.1f}, {np.rad2deg(angles[1]):.1f}, {np.rad2deg(angles[2]):.1f}]")
        if norm:
            print(f"Trans. Norm [cm]: {np.linalg.norm(t_vec) * 100:.2f}")


def _check_dict_key(d, k):
    return k in d and d[k] is not None


def print_metrics(metrics):
    if _check_dict_key(metrics, 'time'):
        print(f"Time:        {metrics['time'] * 1e3:.1f} ms")

    if _check_dict_key(metrics, 't_err'):
        print(f"Trans. Err.: {metrics['t_err'] * 1e2:.1f} cm")
    if _check_dict_key(metrics, 'r_err'):
        print(f"Rot. Err.:   {np.rad2deg(metrics['r_err']):.2f} deg")

    if _check_dict_key(metrics, 'qcqpdq_method'):
        print(f"QCQP Method: {metrics['qcqpdq_method'].name}")
    if _check_dict_key(metrics, 'cost'):
        print(f"Cost:        {metrics['cost']:.2e}")
    if _check_dict_key(metrics, 'gap'):
        if _check_dict_key(metrics, 'gap_rel'):
            print(f"Gap:         {metrics['gap']:.2e} ({np.abs(metrics['gap_rel']) * 1e2:.3f} %)")
        else:
            print(f"Gap:         {metrics['gap']:.2e}")
    if _check_dict_key(metrics, 'conditioning'):
        print(f"Cond. Num.:  {metrics['conditioning']:.2f}")
    if _check_dict_key(metrics, 'n_inliers'):
        print(f"Inliers:     {metrics['n_inliers']}")


def _get_error_metadata(key):
    if key == 'trans':
        def fun(x):
            return x
        label = "Translation [m]"

    elif key == 'rot':
        def fun(x):
            return np.rad2deg(x)
        label = "Rotation [deg]"

    else:
        def fun(x):
            return x
        label = key

    return fun, label


def plot_ransac_errors(errors, method_config, log=False):
    plt.figure()
    plt.suptitle("RANSAC Errors")

    keys = list(errors.dtype.names)
    for idx, key in enumerate(keys):
        # metadata
        err_fun, err_label = _get_error_metadata(key)

        # errors
        plt.subplot(len(keys), 1, idx + 1)
        plt.plot(err_fun(errors[key]))

        plt.xlabel("Sample Index")
        plt.ylabel(err_label)
        if log:
            plt.yscale('log')

        # threshold
        thresh_key = f'{key}_thresh'
        if thresh_key in method_config.init_kwargs:
            threshold = err_fun(method_config.init_kwargs[thresh_key])
            plt.axhline(y=threshold, color='k', linestyle='--')
