from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import hm
from .base import HandEyeCalibrationBase


class MatrixQCQP(HandEyeCalibrationBase):
    """| Certifiably Globally Optimal Extrinsic Calibration from Per-Sensor Egomotion
    | M. Giamou, Z. Ma, V. Peretroukhin, and J. Kelly
    | IEEE Robotics and Automation Letters (Vol. 4, Issue 2), 2019"""

    @staticmethod
    def name():
        return 'MatrixQCQP'

    def __init__(self, normalize=False):
        super().__init__()
        self._Mlist_r = None
        self._Mlist_t = None
        self._Q = None
        self._normalize = normalize

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by MatrixQCQP")
        self._Mlist_r, self._Mlist_t = hm.generation.gen_Mlist(transforms_a, transforms_b, scaled=False)
        self._Q = hm.generation.gen_Q(self._Mlist_r, self._Mlist_t, scaled=False, normalize=self._normalize)

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")
        return hm.optimization.optimize_qcqp(self._Q, **kwargs)


class MatrixQCQPScaled(HandEyeCalibrationBase):
    """| Certifiably Optimal Monocular Hand-Eye Calibration
    | E. Wise, M. Giamou1, S. Khoubyarian, A. Grover, and J. Kelly
    | IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI), 2020"""

    @staticmethod
    def name():
        return 'MatrixQCQPScaled'

    def __init__(self, normalize=False):
        super().__init__()
        self._Mlist_r = None
        self._Mlist_t = None
        self._Q = None
        self._normalize = normalize

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise RuntimeError("Weights are not supported by MatrixQCQP")
        self._Mlist_r, self._Mlist_t = hm.generation.gen_Mlist(transforms_a, transforms_b, scaled=True)
        self._Q = hm.generation.gen_Q(self._Mlist_r, self._Mlist_t, scaled=True, normalize=self._normalize)

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")
        return hm.optimization.optimize_qcqp_scaled(self._Q, **kwargs)
