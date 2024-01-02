from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from . import dq
from .base import HERWCalibrationBase, HERWData
from .dq.generation import QCQPDQCostFun  # noqa
from .separable import SeparableHERWCalibration
from ..base import PairCalibrationResult
from ..utils.fix_translation import fix_translation_herw

from excalibur.optimization.qcqp import generate_quadratic_cost_matrix
from excalibur.utils.logging import logger
from excalibur.utils.motion3d import container_subset


def _select_best_result(results, data_list):
    # check cycle errors for all results
    cycle_errors = [0.0 for _ in results]
    for idx, result in enumerate(results):
        # skip
        if not result.success:
            cycle_errors[idx] = np.inf
            continue

        # iterate data
        for data in data_list:
            calib_x = result.calib.get_x(data.frame_x)
            calib_y = result.calib.get_y(data.frame_y)

            for ta, tb in zip(data.transforms_a, data.transforms_b):
                cycle = (calib_y * tb).inverse() * (ta * calib_x)
                # diff = cycle.asType(m3d.TransformType.kDualQuaternion).toArray() - \
                #        m3d.DualQuaternionTransform().toArray()
                cycle_errors[idx] += cycle.translationNorm()

    # find minimum error and accumulate run times
    best_index = np.argmin(cycle_errors)
    best_result = results[best_index]
    best_result.run_time = np.sum([result.run_time for result in results])

    return best_result


class DualQuaternionQCQPBase(HERWCalibrationBase):
    """| Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation
    | T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    | IEEE Intelligent Vehicles Symposium (IV), 2023"""

    @staticmethod
    def name():
        return 'DualQuaternionQCQPBase'

    def __init__(self, normalize=False, **gen_args):
        super().__init__()
        self._data = None
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
        self.set_transform_data([HERWData(frame_x='', frame_y='', transforms_a=transforms_a, transforms_b=transforms_b,
                                          weights=weights)])
        self._frame_ids = None

    def set_frame_ids(self, frame_ids):
        self._frame_ids = frame_ids

    def set_transform_data(self, data: List[HERWData]) -> None:
        self._data = data
        Mlist, frame_ids, weights = dq.generation.gen_Mlist_multi(data, **self._gen_args)
        self.set_Mlist(Mlist, frame_ids, weights)

    def set_Mlist(self, Mlist, frame_ids=None, weights=None):
        self._Mlist = Mlist
        if frame_ids is not None:
            self._frame_ids = frame_ids
        if isinstance(Mlist[0], list):
            self._Q = [generate_quadratic_cost_matrix(Ml, weights=None, normalize=self._normalize)
                       for Ml in Mlist]
        else:
            self._Q = generate_quadratic_cost_matrix(self._Mlist, weights, normalize=self._normalize)

    def set_Q(self, Q, frame_ids=None):
        self._Mlist = None
        self._frame_ids = frame_ids
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
        if isinstance(self._Q, list):
            results = [dq.optimization.optimize(Q, self._frame_ids, **kwargs) for Q in self._Q]
            result = _select_best_result(results, self._data)
        else:
            result = dq.optimization.optimize(self._Q, self._frame_ids, **kwargs)

        # fix translation
        if result.success and 't_norms' in kwargs and kwargs['t_norms'] is not None:
            if up_approx is not None:
                result = fix_translation_herw(result, self._data, up_approx=up_approx)
            else:
                logger.warning("Approximate up vector is required for fixing the translation")

        return result


class DualQuaternionQCQPSignSampling(HERWCalibrationBase):
    """| Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation
    | T. Wodtko, M. Horn, M. Buchholz, and K. Dietmayer
    | IEEE Intelligent Vehicles Symposium (IV), 2023"""

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
        self.set_transform_data([HERWData(frame_x='', frame_y='', transforms_a=transforms_a, transforms_b=transforms_b,
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
            return dq.sign_ambiguity.calibrate_herw_sign_sampling_multi(
                self._method, self._transform_data, n_reps=self._n_reps, n_samples=self._n_samples)
        else:
            return dq.sign_ambiguity.calibrate_herw_sign_sampling(
                self._method, self._transform_data[0].transforms_a, self._transform_data[0].transforms_b,
                weights=self._transform_data[0].weights, n_reps=self._n_reps, n_samples=self._n_samples)


class DualQuaternionQCQPSeparableInit(HERWCalibrationBase):
    @staticmethod
    def name():
        return 'DualQuaternionQCQPSeparableInit'

    def __init__(self, **kwargs):
        super().__init__()
        self._sep_method = SeparableHERWCalibration(
            hand_eye_name='DualQuaternionQCQP', frame2frame_name='DualQuaternionQCQP')
        self._main_method = DualQuaternionQCQPBase(**kwargs)

    @property
    def Mlist(self):
        return self._main_method.Mlist

    @property
    def Q(self):
        return self._main_method.Q

    @property
    def frame_ids(self):
        return self._main_method.frame_ids

    def configure(self, **kwargs):
        self._sep_method.configure(**kwargs)
        self._main_method.configure(**kwargs)

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self._sep_method.set_transforms(transforms_a, transforms_b, weights)
        self._main_method.set_transforms(transforms_a, transforms_b, weights)

    def set_transform_data(self, data: List[HERWData]) -> None:
        self._sep_method.set_transform_data(data)
        self._main_method.set_transform_data(data)

    def _calibrate(self, **kwargs):
        # check kwargs
        if len(kwargs) > 0:
            logger.warning("Calibration kwargs are not used")

        # separable calibration
        sep_result = self._sep_method.calibrate()
        if not sep_result.success:
            result = PairCalibrationResult()
            result.aux_data = {'separable_result': sep_result}
            return result

        # adjust Mlist of main calibration
        Mlist_norm, z = dq.sign_ambiguity.normalize_Mlist(
            self._main_method.Mlist, sep_result.calib, frame_ids=self._main_method.frame_ids)
        self._main_method.set_Mlist(Mlist_norm)

        # main calibration
        main_result = self._main_method.calibrate(x0=z)

        # adjust result and return
        main_result.run_time += sep_result.run_time
        main_result.aux_data['separable_result'] = sep_result
        return main_result


class DualQuaternionQCQPSeparableRANSACInit(HERWCalibrationBase):
    @staticmethod
    def name():
        return 'DualQuaternionQCQPSeparableRANSACInit'

    def __init__(self, nreps, rot_thresh, trans_thresh, seed=None, **kwargs):
        super().__init__()
        self._sep_method = SeparableHERWCalibration(
            hand_eye_name='HandEyeRANSAC',
            hand_eye_kwargs={
                'method_name': 'DualQuaternionQCQP',
                'nreps': nreps,
                'rot_thresh': rot_thresh,
                'trans_thresh': trans_thresh,
            },
            frame2frame_name='Frame2FrameRANSAC',
            frame2frame_kwargs={
                'method_name': 'DualQuaternionQCQP',
                'nreps': nreps,
                'rot_thresh': rot_thresh,
                'trans_thresh': trans_thresh,
                'nsamples': 3,
            },
        )
        self._seed = seed
        self._rot_thresh = rot_thresh
        self._trans_thresh = trans_thresh
        self._main_method = DualQuaternionQCQPBase(**kwargs)
        self._data = None

    @property
    def Mlist(self):
        return self._main_method.Mlist

    @property
    def Q(self):
        return self._main_method.Q

    @property
    def frame_ids(self):
        return self._main_method.frame_ids

    def configure(self, **kwargs):
        self._sep_method.configure(**kwargs)
        self._main_method.configure(**kwargs)

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        self._sep_method.set_transforms(transforms_a, transforms_b, weights)

    def set_transform_data(self, data: List[HERWData]) -> None:
        self._sep_method.set_transform_data(data)

    def _is_inlier(self, a, b, x, y):
        cycle = (a * x) / (y * b)
        return cycle.rotationNorm() <= self._rot_thresh and cycle.translationNorm() <= self._trans_thresh

    def _select_inliers(self, herw_data, calib):
        x = calib.x if isinstance(calib.x, m3d.TransformInterface) else calib.x.get(herw_data.frame_x)
        y = calib.y if isinstance(calib.y, m3d.TransformInterface) else calib.y.get(herw_data.frame_y)
        inliers = np.where([self._is_inlier(a, b, x, y)
                            for a, b in zip(herw_data.transforms_a, herw_data.transforms_b)])[0]
        return HERWData(frame_x=herw_data.frame_x, frame_y=herw_data.frame_y,
                        transforms_a=container_subset(herw_data.transforms_a, inliers),
                        transforms_b=container_subset(herw_data.transforms_b, inliers))

    def _calibrate(self, **kwargs):
        # check kwargs
        if len(kwargs) > 0:
            logger.warning("Calibration kwargs are not used")

        # seed
        if self._seed is not None:
            np.random.seed(self._seed)

        # separable calibration
        sep_result = self._sep_method.calibrate()
        if not sep_result.success:
            result = PairCalibrationResult()
            result.aux_data = {'separable_result': sep_result}
            return result

        # select inliers for main calibration based on separable solution
        data = [
            self._select_inliers(herw_data, sep_result.calib)
            for herw_data in self._sep_method.transform_data
        ]
        num_inliers = np.sum([len(herw_data.transforms_a) for herw_data in data])
        self._main_method.set_transform_data(data)
        if not self._sep_method.is_multi:
            self._main_method.set_frame_ids(None)

        # adjust Mlist of main calibration
        Mlist_norm, z = dq.sign_ambiguity.normalize_Mlist(
            self._main_method.Mlist, sep_result.calib, frame_ids=self._main_method.frame_ids)
        self._main_method.set_Mlist(Mlist_norm)

        # main calibration
        main_result = self._main_method.calibrate(x0=z)

        # adjust result and return
        main_result.run_time += sep_result.run_time
        main_result.aux_data['separable_result'] = sep_result
        main_result.aux_data['num_inliers'] = num_inliers
        return main_result
