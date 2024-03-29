from dataclasses import dataclass, field
from typing import Dict, Optional

import excalibur.calibration


@dataclass
class MethodConfig:
    name: str
    kwargs: Optional[Dict] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


def calibrate(method_config: MethodConfig, transforms1, transforms2, calib_gt, **kwargs):
    # calibrate
    method = excalibur.calibration.HandEyeCalibrationBase.create(method_config.name)
    method.configure(**method_config.kwargs)
    method.set_transforms(transforms1, transforms2)
    result = method.calibrate()

    if not result.success:
        print(f"[{method_config.name}] {result.message}")
        if result.calib is None:
            return {'t_err': None,
                    'r_err': None,
                    'time': None,
                    'is_global': None}

    # compare
    calib_pred = result.calib
    calib_error = calib_pred.inverse() * calib_gt

    # globality
    is_global = result.aux_data['is_global'] if 'is_global' in result.aux_data else None
    trans_cond = result.aux_data['conditioning'].trans_cond if 'conditioning' in result.aux_data else None

    return {'t_err': calib_error.translationNorm(),
            'r_err': calib_error.rotationNorm(),
            'time': result.run_time,
            'is_global': is_global,
            'trans_cond': trans_cond}


def calibrate_planar(method_config: MethodConfig, transforms1, transforms2, plane1, plane2, calib_gt):
    # calibrate
    method = excalibur.calibration.HandEyeCalibrationBase.create(method_config.name)
    method.configure(**method_config.kwargs)
    method.set_plane_transforms(plane1, plane2)
    method.set_transforms(transforms1, transforms2)
    result = method.calibrate()

    if not result.success:
        print(f"[{method_config.name}] {result.message}")
        if result.calib is None:
            return {'t_err': None,
                    'r_err': None,
                    'time': None,
                    'is_global': None}

    # globality
    is_global = result.aux_data['is_global'] if 'is_global' in result.aux_data else None

    # compare
    calib_pred = result.calib
    calib_error = calib_pred.inverse() * calib_gt

    return {'t_err': calib_error.translationNorm(),
            'r_err': calib_error.rotationNorm(),
            'time': result.run_time,
            'is_global': is_global}
