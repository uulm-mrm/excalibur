from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HERWCalibrationBase, HERWData


class Wang(HERWCalibrationBase):
    """| Accurate Calibration of Multi-Perspective Cameras from a Generalization of the Hand-Eye Constraint
    | Y. Wang, W. Jiang, K. Huang, S. Schwertfeger, and L. Kneip
    | IEEE International Conference on Robotics and Automation (ICRA), 2022"""

    @staticmethod
    def name():
        return 'Wang'

    def __init__(self):
        super().__init__()
        self._wang_data = None
        self._frame_ids = None

    def set_transforms(self, transforms_a: Union[m3d.TransformContainer, List],
                       transforms_b: Union[m3d.TransformContainer, List],
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Wang")
        self.set_transform_data([HERWData(frame_x='', frame_y='',
                                          transforms_a=transforms_a, transforms_b=transforms_b)])
        self._frame_ids = None

    def set_transform_data(self, data: List[HERWData]) -> None:
        self._wang_data, self._frame_ids = hm.generation.gen_wang(data)

    def _calibrate(self, **kwargs):
        if self._wang_data is None:
            raise RuntimeError("Linear formulation is missing")
        return hm.analytic.solve_wang(self._wang_data, self._frame_ids, **kwargs)
