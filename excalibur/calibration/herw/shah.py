from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HERWCalibrationBase


class Shah(HERWCalibrationBase):
    """| Solving the robot-world/hand-eye calibration problem using the kronecker product
    | M. Shah
    | Journal of Mechanisms and Robotics (Vol. 5, Issue 3), 2013"""

    @staticmethod
    def name():
        return 'Shah'

    def __init__(self, **kwargs):
        super().__init__()
        self._T = None
        self._A = None
        self._b_data = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Shah")
        self._T, self._A, self._b_data = hm.generation.gen_shah(transforms_a, transforms_b)

    def _calibrate(self, **_):
        if self._T is None or self._A is None or self._b_data is None:
            raise RuntimeError("Linear formulation is missing")
        return hm.analytic.solve_shah(self._T, self._A, self._b_data)
