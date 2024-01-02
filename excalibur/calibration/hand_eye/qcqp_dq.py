from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from excalibur.optimization.qcqp import generate_quadratic_cost_matrix
from excalibur.utils.logging import logger

from . import dq
from .base import HandEyeCalibrationBase
from ..utils.fix_translation import fix_translation_hand_eye
from ..utils.conditioning import get_conditioning


class DualQuaternionQCQP(HandEyeCalibrationBase):
    """| Online Extrinsic Calibration based on Per-Sensor Ego-Motion Using Dual Quaternions
    | M. Horn, T. Wodtko, M. Buchholz, and K. Dietmayer
    | IEEE Robotics and Automation Letters (Vol. 6, Issue 2), 2021"""

    @staticmethod
    def name():
        return 'DualQuaternionQCQP'

    def __init__(self, normalize: bool = False):
        super().__init__()
        self._transforms_a = None
        self._Mlist = None
        self._Q = None
        self._normalize = normalize

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self._transforms_a = transforms_a
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

        # process kwargs
        up_approx = None
        if 'up_approx' in kwargs:
            up_approx = kwargs['up_approx']
            del kwargs['up_approx']

        # optimize
        result = dq.optimization.optimize_qcqp(self._Q, **kwargs)

        # post-processing
        if result.success:
            if 't_norm' in kwargs:
                # fix translation
                if up_approx is not None:
                    result = fix_translation_hand_eye(result, self._transforms_a, up_approx=up_approx)
                else:
                    logger.warning("Approximate up vector is required for fixing the translation")
            elif result.aux_data['is_global']:
                # conditioning
                result.aux_data['conditioning'] = get_conditioning(self._Q, result.calib)

        return result


class DualQuaternionQCQPPlanar(DualQuaternionQCQP):
    """| Online Extrinsic Calibration based on Per-Sensor Ego-Motion Using Dual Quaternions
    | M. Horn, T. Wodtko, M. Buchholz, and K. Dietmayer
    | IEEE Robotics and Automation Letters (Vol. 6, Issue 2), 2021"""

    @staticmethod
    def name():
        return 'DualQuaternionQCQPPlanar'

    def __init__(self, normalize: bool = False):
        super().__init__(normalize=normalize)
        self.configure(planar_only=True)
        self._plane_a = None
        self._plane_b = None

    def configure(self, **kwargs):
        if 'plane_a' in kwargs and 'plane_b' in kwargs:
            self.set_plane_transforms(kwargs['plane_a'], kwargs['plane_b'])
            del kwargs['plane_a']
            del kwargs['plane_b']
        elif ('plane_a' in kwargs and 'plane_b' not in kwargs) or ('plane_a' not in kwargs and 'plane_b' in kwargs):
            raise RuntimeError("Please provide plane_a and plane_b")
        super().configure(**kwargs)

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
    """| Globally Optimal Multi-Scale Monocular Hand-Eye Calibration Using Dual Quaternions
    | T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    | International Conference on 3D Vision (3DV), 2021"""

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
