from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import HandEyeCalibrationBase
from excalibur.optimization.qcqp import generate_quadratic_cost_matrix


class DualQuaternionQCQP(HandEyeCalibrationBase):
    # Online Extrinsic Calibration based on Per-Sensor Ego-Motion Using Dual Quaternions
    # M. Horn, T. Wodtko, M. Buchholz, and K. Dietmayer
    # IEEE Robotics and Automation Letters (Vol. 6, Issue 2), 2021

    @staticmethod
    def name():
        return 'DualQuaternionQCQP'

    def __init__(self, normalize: bool = False):
        super().__init__()
        self._Mlist = None
        self._Q = None
        self._normalize = normalize

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self._Mlist = dq.generation.gen_Mlist(transforms_a, transforms_b, daniilidis=False)
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Mlist(self, Mlist, weights=None):
        self._Mlist = Mlist
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q):
        self._Mlist = None
        self._Q = Q

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")
        return dq.optimization.optimize_qcqp(self._Q, **kwargs)


class DualQuaternionQCQPPlanar(DualQuaternionQCQP):
    # Online Extrinsic Calibration based on Per-Sensor Ego-Motion Using Dual Quaternions
    # M. Horn, T. Wodtko, M. Buchholz, and K. Dietmayer
    # IEEE Robotics and Automation Letters (Vol. 6, Issue 2), 2021

    @staticmethod
    def name():
        return 'DualQuaternionQCQPPlanar'

    def __init__(self, normalize: bool = False):
        super().__init__(normalize=normalize)
        self.configure(planar_only=True)
        self._plane_a = None
        self._plane_b = None

    def set_plane_transforms(self, plane_a: m3d.TransformInterface, plane_b: m3d.TransformInterface):
        self._plane_a = plane_a.copy()
        self._plane_b = plane_b.copy()

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if self._plane_a is None or self._plane_b is None:
            raise RuntimeError("Planes must be provided before setting the transforms.")

        # adjust transforms
        transforms_a_planar = transforms_a.changeFrame(self._plane_a.inverse()).normalized_()
        transforms_b_planar = transforms_b.changeFrame(self._plane_b.inverse()).normalized_()

        # pass to parent class
        super().set_transforms(transforms_a_planar, transforms_b_planar, weights)

    def _calibrate(self, **kwargs):
        if self._plane_a is None or self._plane_b is None:
            raise RuntimeError("Planes must be provided before calibrating.")
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")

        # calibrate
        result = super()._calibrate(**kwargs)

        # adjust transform and return
        result.calib = (self._plane_a.inverse() * result.calib * self._plane_b).normalized_()
        return result


class DualQuaternionQCQPScaled(HandEyeCalibrationBase):
    # Globally Optimal Multi-Scale Monocular Hand-Eye Calibration Using Dual Quaternions
    # T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    # International Conference on 3D Vision (3DV), 2021

    @staticmethod
    def name():
        return 'DualQuaternionQCQPScaled'

    def __init__(self, normalize: bool = False):
        super().__init__()
        self._Mlist = None
        self._Q = None
        self._normalize = normalize

    def set_transforms(self, transforms_a, transforms_b,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        # handle inputs
        if isinstance(transforms_a, (list, tuple)) and isinstance(transforms_b, (list, tuple)):
            assert len(transforms_a) == len(transforms_b)
        else:
            transforms_a = [transforms_a]
            transforms_b = [transforms_b]

        self._Mlist = dq.generation.gen_Mlist_scaled(transforms_a, transforms_b, daniilidis=False)
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Mlist(self, Mlist, weights=None):
        self._Mlist = Mlist
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q):
        self._Mlist = None
        self._Q = Q

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")
        return dq.optimization.optimize_qcqp_scaled(self._Q, **kwargs)
