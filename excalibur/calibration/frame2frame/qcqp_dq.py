from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import Frame2FrameCalibrationBase
from excalibur.optimization.qcqp import generate_quadratic_cost_matrix


class DualQuaternionQCQP(Frame2FrameCalibrationBase):
    @staticmethod
    def name():
        return 'DualQuaternionQCQP'

    def __init__(self, normalize=False):
        super().__init__()
        self._Mlist = None
        self._Q = None
        self._normalize = normalize

    @property
    def Mlist(self):
        return self._Mlist

    @property
    def Q(self):
        return self._Q

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None, **kwargs) -> None:
        self._Mlist = dq.generation.gen_Mlist(transforms_a, transforms_b, **kwargs)
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Mlist(self, Mlist, weights=None):
        self._Mlist = Mlist
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q):
        self._Mlist = None
        self._Q = Q

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q is missing")
        return dq.optimization.optimize(self._Q, **kwargs)
