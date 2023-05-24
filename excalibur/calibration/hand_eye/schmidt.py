from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq, hm
from .base import HandEyeCalibrationBase


class SchmidtDQ(HandEyeCalibrationBase):
    # Calibration-Free Hand-Eye Calibration: A Structure-from-Motion Approach
    # J. Schmidt, F. Vogt, and H. Niemann
    # Pattern Recognition, 2005

    @staticmethod
    def name():
        return 'SchmidtDQ'

    def __init__(self):
        super().__init__()
        self._data = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Schmidt")
        self._data = dq.generation.gen_schmidt(transforms_a, transforms_b)

    def _calibrate(self, **kwargs):
        if self._data is None:
            raise RuntimeError("Transformation data are is missing")
        return dq.optimization.optimize_schmidt(self._data, **kwargs)


class SchmidtHM(HandEyeCalibrationBase):
    # Calibration-Free Hand-Eye Calibration: A Structure-from-Motion Approach
    # J. Schmidt, F. Vogt, and H. Niemann
    # Pattern Recognition, 2005

    @staticmethod
    def name():
        return 'SchmidtHM'

    def __init__(self):
        super().__init__()
        self._data = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Schmidt")
        self._data = hm.generation.gen_schmidt(transforms_a, transforms_b)

    def _calibrate(self, **kwargs):
        if self._data is None:
            raise RuntimeError("Transformation data are is missing")
        return hm.optimization.optimize_schmidt(self._data, **kwargs)
