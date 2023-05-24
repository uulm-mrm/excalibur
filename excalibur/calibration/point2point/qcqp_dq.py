from typing import List, Optional, Union

import numpy as np

from . import dq
from .base import Point2PointCalibrationBase
from excalibur.optimization.qcqp import generate_quadratic_cost_matrix


class DualQuaternionQCQP(Point2PointCalibrationBase):
    @staticmethod
    def name():
        return 'DualQuaternionQCQP'

    def __init__(self, normalize=False):
        super().__init__()
        self._Mlist = None
        self._Q = None
        self._normalize = normalize

    def set_points(self, points_a: np.ndarray, points_b: np.ndarray,
                   weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self._Mlist = dq.generation.gen_Mlist(points_a, points_b)
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Mlist(self, Mlist, weights):
        self._Mlist = Mlist
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q):
        self._Mlist = None
        self._Q = Q

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q is missing")
        return dq.optimization.optimize(self._Q, **kwargs)
