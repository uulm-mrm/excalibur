from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HERWCalibrationBase


class Dornaika(HERWCalibrationBase):
    # Simultaneous Robot-World and Hand-Eye Calibration
    # F. Dornaika and R. Horaud
    # IEEE Transactions on Robotics and Automation (Vol. 14, Issue 4), 1998

    @staticmethod
    def name():
        return 'Dornaika'

    def __init__(self):
        super().__init__()
        self._matrix_data = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Dornaika")
        self._matrix_data = hm.generation.gen_matrix_data(transforms_a, transforms_b)

    def _calibrate(self, **kwargs):
        if self._matrix_data is None:
            raise RuntimeError("Transformations are missing")
        return hm.optimization.optimize_dornaika(self._matrix_data, **kwargs)
