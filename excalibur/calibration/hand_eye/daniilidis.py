from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import HandEyeCalibrationBase


class Daniilidis(HandEyeCalibrationBase):
    """| Hand-Eye Calibration Using Dual Quaternions
    | K. Daniilidis
    | The International Journal of Robotics Research (Vol. 18, Issue 3), 1999"""

    @staticmethod
    def name():
        return 'Daniilidis'

    def __init__(self):
        super().__init__()
        self._Mlist = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Daniilidis")
        self._Mlist = dq.generation.gen_Mlist(transforms_a, transforms_b, daniilidis=True)

    def set_Mlist(self, Mlist):
        self._Mlist = Mlist

    def _calibrate(self, **_):
        if self._Mlist is None:
            raise RuntimeError("M list is missing")
        return dq.analytic.solve_svd(self._Mlist)
