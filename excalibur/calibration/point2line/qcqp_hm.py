from typing import List, Optional, Union

import numpy as np

from excalibur.fitting.line import Line

from . import hm
from .base import Point2LineCalibrationBase


class MatrixQCQP(Point2LineCalibrationBase):
    """| Convex Global 3D Registration with Lagrangian Duality
    | J. Briales and J. Gonzalez-Jimenez
    | IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017"""

    @staticmethod
    def name():
        return 'MatrixQCQP'

    def __init__(self, normalize=False):
        super().__init__()
        self._Q = None
        self._normalize = normalize

    def set_data(self, points_a: np.ndarray, lines_b: List[Line],
                 weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by MatrixQCQP")
        self._Q = hm.generation.gen_Q(points_a, lines_b, self._normalize)

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q is missing")
        return hm.optimization.optimize(self._Q, **kwargs)
