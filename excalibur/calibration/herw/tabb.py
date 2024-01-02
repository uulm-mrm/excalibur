from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HERWCalibrationBase, HERWData


class Tabb(HERWCalibrationBase):
    """| Solving the robot-world hand-eye(s) calibration problem with iterative methods
    | A. Tabb and K. M. A. Yousef
    | Machine Vision and Applications (Vol. 28, Issue 5), 2017"""

    @staticmethod
    def name():
        return 'Tabb'

    def __init__(self):
        super().__init__()
        self._matrix_data_list = None
        self._frame_ids = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Tabb")
        self.set_transform_data([HERWData(frame_x='', frame_y='',
                                          transforms_a=transforms_a, transforms_b=transforms_b)])
        self._frame_ids = None

    def set_transform_data(self, data: List[HERWData]) -> None:
        for d in data:
            if d.weights is not None:
                raise RuntimeError("Weights are not supported by Tabb")
        self._matrix_data_list, self._frame_ids = hm.generation.gen_matrix_data_multi(data)

    def _calibrate(self, **kwargs):
        if self._matrix_data_list is None:
            raise RuntimeError("Transformations are is missing")
        return hm.optimization.optimize_tabb(self._matrix_data_list, self._frame_ids, **kwargs)
