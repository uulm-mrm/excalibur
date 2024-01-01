from typing import List, Optional, Union

import numpy as np

from . import quat
from .base import Point2PointCalibrationBase


class HornQuat(Point2PointCalibrationBase):
    """| Closed-form solution of absolute orientation using unit quaternions
    | B. K. P. Horn
    | Journal of the Optical Society of America A (Vol. 4), 1987"""

    @staticmethod
    def name():
        return 'HornQuat'

    def __init__(self):
        super().__init__()
        self._points_a = None
        self._points_b = None

    def set_points(self, points_a: np.ndarray, points_b: np.ndarray,
                   weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by Arun")
        if points_a.shape[0] != 3 or points_b.shape[0] != 3 or points_a.shape[1] != points_b.shape[1]:
            raise RuntimeError("Both point clouds must have shapes (3, n)")
        self._points_a = points_a
        self._points_b = points_b

    def _calibrate(self, **_):
        if self._points_a is None or self._points_b is None:
            raise RuntimeError("Point data is missing")
        return quat.analytic.solve_horn_quat(self._points_a, self._points_b)
