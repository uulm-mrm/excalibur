from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import HERWCalibrationBase, HERWData
from excalibur.optimization.qcqp import generate_quadratic_cost_matrix


class DualQuaternionQCQPBase(HERWCalibrationBase):
    # Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation
    # T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    # IEEE Intelligent Vehicles Symposium (IV), 2023

    @staticmethod
    def name():
        return 'DualQuaternionQCQPBase'

    def __init__(self, normalize=False, **gen_args):
        super().__init__()
        self._Mlist = None
        self._Q = None
        self._frame_ids = None
        self._normalize = normalize
        self._gen_args = gen_args

    @property
    def Mlist(self):
        return self._Mlist

    @property
    def Q(self):
        return self._Q

    @property
    def frame_ids(self):
        return self._frame_ids

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self.set_transform_data([HERWData(frame_x=0, frame_y=0, transforms_a=transforms_a, transforms_b=transforms_b,
                                          weights=weights)])
        self._frame_ids = None

    def set_frame_ids(self, frame_ids):
        self._frame_ids = frame_ids

    def set_transform_data(self, data: List[HERWData]) -> None:
        Mlist, frame_ids, weights = dq.generation.gen_Mlist_multi(data, **self._gen_args)
        self.set_Mlist(Mlist, frame_ids, weights)

    def set_Mlist(self, Mlist, frame_ids=None, weights=None):
        self._Mlist = Mlist
        if frame_ids is not None:
            self._frame_ids = frame_ids
        self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q, frame_ids=None):
        self._Mlist = None
        self._frame_ids = frame_ids
        self._Q = Q

    def _calibrate(self, **kwargs):
        if self._Q is None:
            raise RuntimeError("Q matrix is missing")
        return dq.optimization.optimize(self._Q, self._frame_ids, **kwargs)


class DualQuaternionQCQPSignSampling(HERWCalibrationBase):
    # Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation
    # T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    # IEEE Intelligent Vehicles Symposium (IV), 2023

    @staticmethod
    def name():
        return 'DualQuaternionQCQPSignSampling'

    def __init__(self, n_iter=1, n_samples=3, **kwargs):
        super().__init__()
        self._n_reps = n_iter
        self._n_samples = n_samples
        self._method = DualQuaternionQCQPBase(**kwargs)
        self._transform_data = None
        self._is_multi = False

    @property
    def Mlist(self):
        return self._method.Mlist

    @property
    def Q(self):
        return self._method.Q

    @property
    def frame_ids(self):
        return self._method.frame_ids

    def configure(self, **kwargs):
        self._method.configure(**kwargs)

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self.set_transform_data([HERWData(frame_x=0, frame_y=0, transforms_a=transforms_a, transforms_b=transforms_b,
                                          weights=weights)])
        self._is_multi = False

    def set_transform_data(self, data: List[HERWData]) -> None:
        self._transform_data = data
        self._is_multi = True

    def _calibrate(self, **kwargs):
        if self._transform_data is None:
            raise RuntimeError("Transformation data are missing")
        self._method.configure(**kwargs)

        if self._is_multi:
            return dq.sign_sampling.calibrate_herw_sign_sampling_multi(
                self._method, self._transform_data, n_reps=self._n_reps, n_samples=self._n_samples)
        else:
            return dq.sign_sampling.calibrate_herw_sign_sampling(
                self._method, self._transform_data[0].transforms_a, self._transform_data[0].transforms_b,
                weights=self._transform_data[0].weights, n_reps=self._n_reps, n_samples=self._n_samples)
