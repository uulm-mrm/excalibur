from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HandEyeCalibrationBase


class Andreff(HandEyeCalibrationBase):
    # Robot Hand-Eye Calibration using Structure from Motion
    # N. Andreff, R. Horaud, and B. Espiau
    # The International Journal of Robotics Research (Vol. 20, Issue 3), 2001

    @staticmethod
    def name():
        return 'Andreff'

    def __init__(self):
        super().__init__()
        self._A = None
        self._b = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Andreff")
        self._A, self._b = hm.generation.gen_linear_andreff(transforms_a, transforms_b)

    def _calibrate(self, **_):
        if self._A is None or self._b is None:
            raise RuntimeError("Linear formulation is missing")
        return hm.analytic.solve_linear(self._A, self._b)
