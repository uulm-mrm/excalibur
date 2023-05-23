from typing import List, Optional, Union

import numpy as np

from . import hm
from .base import Point2PointCalibrationBase


class Arun(Point2PointCalibrationBase):
    # Least-Squares Fitting of Two 3-D Point Sets
    # K. S. Arun, T. S. Huang, and S.D. Blostein
    # IEEE Transactions on Pattern Analysis and Machine Intelligence, 1987

    @staticmethod
    def name():
        return 'Arun'

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
        return hm.analytic.solve_arun(self._points_a, self._points_b)
