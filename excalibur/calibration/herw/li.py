from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq, hm
from .base import HERWCalibrationBase


class LiDQBase(HERWCalibrationBase):
    # Simultaneous robot-world and hand-eye calibration using dual-quaternions and Kronecker product
    # A. Li, L. Wang and D. Wu
    # International Journal of the Physical Sciences (Vol. 5, Issue 10), 2010

    @staticmethod
    def name():
        return 'LiDQBase'

    def __init__(self):
        super().__init__()
        self._Mlist = None

    @property
    def Mlist(self):
        return self._Mlist

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by LiDQ")
        self.set_Mlist(dq.generation.gen_Mlist(transforms_a, transforms_b))

    def set_Mlist(self, Mlist, weights=None):
        if weights is not None:
            raise RuntimeError("Weights are not supported by LiDQ")
        self._Mlist = Mlist

    def _calibrate(self, **_):
        if self._Mlist is None:
            raise RuntimeError("Mlist is missing")
        return dq.analytic.solve_svd(self._Mlist)


class LiDQSignSampling(HERWCalibrationBase):
    # Simultaneous robot-world and hand-eye calibration using dual-quaternions and Kronecker product
    # A. Li, L. Wang and D. Wu
    # International Journal of the Physical Sciences (Vol. 5, Issue 10), 2010

    @staticmethod
    def name():
        return 'LiDQSignSampling'

    def __init__(self, n_iter=1, n_samples=3):
        super().__init__()
        self._n_reps = n_iter
        self._n_samples = n_samples
        self._method = LiDQBase()
        self._transforms_a = None
        self._transforms_b = None

    def configure(self, **kwargs):
        self._method.configure(**kwargs)

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by LiDQ")
        self._transforms_a = transforms_a
        self._transforms_b = transforms_b

    def _calibrate(self, **kwargs):
        if self._transforms_a is None:
            raise RuntimeError("Transformations are missing")
        self._method.configure(**kwargs)
        return dq.sign_sampling.calibrate_herw_sign_sampling(
            self._method, self._transforms_a, self._transforms_b, n_reps=self._n_reps, n_samples=self._n_samples)


class LiHM(HERWCalibrationBase):
    # Simultaneous robot-world and hand-eye calibration using dual-quaternions and Kronecker product
    # A. Li, L. Wang and D. Wu
    # International Journal of the Physical Sciences (Vol. 5, Issue 10), 2010

    @staticmethod
    def name():
        return 'LiHM'

    def __init__(self):
        super().__init__()
        self._A = None
        self._b = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by LiHM")
        self._A, self._b = hm.generation.gen_linear_li(transforms_a, transforms_b)

    def _calibrate(self, **_):
        if self._A is None or self._b is None:
            raise RuntimeError("Linear formulation is missing")
        return hm.analytic.solve_linear(self._A, self._b)
