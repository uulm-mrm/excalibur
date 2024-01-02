from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import HandEyeCalibrationBase


class Wei(HandEyeCalibrationBase):
    """| Calibration-Free Robot-Sensor Calibration approach based on Second-Order Cone Programming
    | L. Wei, L. Naiguang, D. Mingli, and L. Xiaoping
    | MATEC Web of Conferences, 2018"""

    @staticmethod
    def name():
        return 'Wei'

    def __init__(self):
        super().__init__()
        self._data = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Wei")
        self._data = dq.generation.gen_wei(transforms_a, transforms_b)

    def _calibrate(self, **_):
        if self._data is None:
            raise RuntimeError("Transformation data are is missing")
        return dq.optimization.optimize_wei(self._data)
