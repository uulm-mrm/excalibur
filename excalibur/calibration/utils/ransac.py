from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import List, Optional, Union

import motion3d as m3d
import numpy as np
import pandas as pd

from excalibur.utils.motion3d import container_subset
from excalibur.utils.logging import LogLevelContext

from ..base import CalibrationResult, CalibrationResultScaled
from ..frame2frame.base import Frame2FrameCalibrationBase
from ..hand_eye.base import HandEyeCalibrationBase
from ..herw.base import HERWCalibrationBase, HERWData
from ..point2point.base import Point2PointCalibrationBase


DEFAULT_ITER_LOG_LEVEL = logging.FATAL


class BaseRANSAC(ABC):
    def __init__(self):
        super().__init__()

    def _initialize(self, method_name, nreps, rot_thresh=None, trans_thresh=None,
                    nsamples=None, seed=None, iter_log_level=DEFAULT_ITER_LOG_LEVEL, **kwargs):
        self._method = self._init_method(method_name, **kwargs)
        self._nreps = nreps

        self._rot_thresh = rot_thresh
        self._trans_thresh = trans_thresh

        self._nsamples = nsamples if nsamples is not None else self._min_samples
        self._seed = seed
        self._iter_log_level = iter_log_level
        self._data = None

    @staticmethod
    @abstractmethod
    def _init_method(method_name, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def _min_samples(self) -> int:
        raise NotImplementedError

    def _data_size(self):
        return len(self._data[0])

    def _draw_sample_indices(self, _rep: int):
        return np.random.choice(np.arange(self._data_size()), self._nsamples, replace=False)

    def _iter_samples(self):
        for a, b in zip(self._data[0], self._data[1]):
            yield a, b

    def _set_subset(self, indices):
        if len(indices) < self._min_samples:
            return False
        data_a = container_subset(self._data[0], indices)
        data_b = container_subset(self._data[1], indices)
        self._method.set_transforms(data_a, data_b)
        return True

    def _get_inliers(self, result):
        return np.where([self._is_inlier(a, b, result) for a, b in self._iter_samples()])[0]

    def _inliers_size(self, inliers):
        return len(inliers)

    @abstractmethod
    def _get_sample_errors(self, a, b, result):
        raise NotImplementedError

    def _get_data_errors(self, result):
        errors = [self._get_sample_errors(a, b, result) for a, b in self._iter_samples()]
        return pd.DataFrame(errors).to_records(index=False)

    @abstractmethod
    def _is_inlier(self, a, b, result):
        raise NotImplementedError

    def configure(self, **kwargs):
        self._method.configure(**kwargs)

    def _calibrate(self, **kwargs):
        # check data
        if self._data is None:
            raise RuntimeError("Data are missing")

        # seed
        if self._seed is not None:
            np.random.seed(self._seed)

        # repeat multiple times
        best_inliers = []
        best_num_inliers = 0
        best_rep = None
        run_time = 0.0
        for rep in range(self._nreps):
            # random indices
            sample_indices = self._draw_sample_indices(rep)

            # calibrate
            if not self._set_subset(sample_indices):
                continue

            with LogLevelContext(self._iter_log_level):
                rep_result = self._method.calibrate(**kwargs)

            if rep_result.run_time is not None:
                run_time += rep_result.run_time
            if not rep_result.success:
                continue

            # count inliers
            inliers = self._get_inliers(rep_result)
            if self._inliers_size(inliers) > best_num_inliers:
                best_inliers = inliers
                best_num_inliers = self._inliers_size(inliers)
                best_rep = rep
                if self._inliers_size(best_inliers) == self._data_size():
                    break

        # repeat calibration with best inliers
        if self._set_subset(best_inliers):
            result = self._method.calibrate(**kwargs)
            if not result.success:
                return result
            result.run_time += run_time
            result.aux_data['inliers'] = best_inliers
            result.aux_data['best_rep'] = best_rep
        else:
            result = CalibrationResult()
            return result

        # final errors
        if result.calib is not None:
            result.aux_data['errors'] = self._get_data_errors(result)

        return result


class Point2PointRANSAC(BaseRANSAC, Point2PointCalibrationBase):
    @staticmethod
    def name():
        return 'Point2PointRANSAC'

    def __init__(self, *args, **kwargs):
        super(BaseRANSAC, self).__init__()
        super(Point2PointCalibrationBase, self).__init__()
        self._initialize(*args, **kwargs)
        assert self._trans_thresh is not None

    def set_points(self, points_a: m3d.TransformContainer, points_b: m3d.TransformContainer,
                   weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self._data = (points_a, points_b)

    def set_transform_data(self, data) -> None:
        raise NotImplementedError("Multiple transforms are not supported by RANSAC")

    @staticmethod
    def _init_method(method_name, **kwargs):
        return Point2PointCalibrationBase.create(method_name, **kwargs)

    @property
    def _min_samples(self) -> int:
        return 3

    def _data_size(self):
        return self._data[0].shape[1]

    def _iter_samples(self):
        for col in range(self._data[0].shape[1]):
            yield self._data[0][:, col], self._data[1][:, col]

    def _set_subset(self, indices):
        if len(indices) < self._min_samples:
            return False
        data_a = self._data[0][:, indices]
        data_b = self._data[1][:, indices]
        self._method.set_points(data_a, data_b)
        return True

    def _get_sample_errors(self, a, b, result):
        calib = result.calib
        if len(a) == 2:
            a_3d = np.array([*a, 0.0])
            b_3d = np.array([*b, 0.0])
            return {'trans': np.linalg.norm(a_3d - calib.transformPoint(b_3d))}
        else:
            return {'trans': np.linalg.norm(a - calib.transformPoint(b))}

    def _is_inlier(self, a, b, result):
        errors = self._get_sample_errors(a, b, result)
        return errors['trans'] <= self._trans_thresh


class Frame2FrameRANSAC(BaseRANSAC, Frame2FrameCalibrationBase):
    @staticmethod
    def name():
        return 'Frame2FrameRANSAC'

    def __init__(self, *args, **kwargs):
        super(BaseRANSAC, self).__init__()
        super(Frame2FrameCalibrationBase, self).__init__()
        self._initialize(*args, **kwargs)
        assert self._rot_thresh is not None and self._trans_thresh is not None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self._data = (transforms_a, transforms_b)

    def set_transform_data(self, data) -> None:
        raise NotImplementedError("Multiple transforms are not supported by RANSAC")

    @staticmethod
    def _init_method(method_name, **kwargs):
        return Frame2FrameCalibrationBase.create(method_name, **kwargs)

    @property
    def _min_samples(self) -> int:
        return 1

    def _get_sample_errors(self, a, b, result):
        cycle = a / (result.calib * b)
        return {'rot': cycle.rotationNorm(), 'trans': cycle.translationNorm()}

    def _is_inlier(self, a, b, result):
        errors = self._get_sample_errors(a, b, result)
        return errors['rot'] <= self._rot_thresh and errors['trans'] <= self._trans_thresh


class BaseRANSACWeighted(BaseRANSAC, ABC):
    def __init__(self):
        super().__init__()

    def _initialize(self, method_name, nreps, rot_thresh=None, trans_thresh=None, nth_weighted=0, nsamples=None,
                    seed=None, iter_log_level=DEFAULT_ITER_LOG_LEVEL, **kwargs):
        super()._initialize(method_name, nreps, rot_thresh=rot_thresh, trans_thresh=trans_thresh, nsamples=nsamples,
                            seed=seed, iter_log_level=iter_log_level, **kwargs)
        self._nth_weighted = nth_weighted

    def _draw_sample_indices(self, rep: int):
        # create index list
        available_indices = np.arange(self._data_size())

        if self._nsamples == 1 or self._nth_weighted == 0 or rep % self._nth_weighted != 0:
            # uniform sampling
            return np.random.choice(available_indices, self._nsamples, replace=False)
        else:
            # draw all samples except the last uniformly
            samples = np.random.choice(available_indices, self._nsamples - 1, replace=False)
            available_indices = np.delete(available_indices, samples)
            assert len(available_indices) == self._data_size() - self._nsamples + 1

            # get sample weights
            first_axis = self._data[0][samples[0]].asType(m3d.TransformType.kAxisAngle).getAxis()
            sample_weights = np.array([
                1.0 - np.abs(self._data[0][idx].asType(m3d.TransformType.kAxisAngle).getAxis().T @ first_axis)
                for idx in available_indices
            ])
            sample_weights[sample_weights < 1e-6] = 0.0

            # normalize sample weights and draw last sample
            sample_weights_sum = np.sum(sample_weights)
            if sample_weights_sum < 1e-6:
                last_sample = np.random.choice(available_indices, 1)
            else:
                sample_weights /= sample_weights_sum
                last_sample = np.random.choice(available_indices, 1, p=sample_weights)

            # combine and return
            samples = np.concatenate((samples, last_sample))
            return samples


class HandEyeRANSAC(BaseRANSACWeighted, HandEyeCalibrationBase):
    @staticmethod
    def name():
        return 'HandEyeRANSAC'

    def __init__(self, *args, **kwargs):
        super(BaseRANSACWeighted, self).__init__()
        super(HandEyeCalibrationBase, self).__init__()
        self._initialize(*args, **kwargs)
        assert self._rot_thresh is not None and self._trans_thresh is not None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self._data = (transforms_a, transforms_b)

    def set_transform_data(self, data) -> None:
        raise NotImplementedError("Multiple transforms are not supported by RANSAC")

    def set_plane_transforms(self, *args):
        self._method.set_plane_transforms(*args)

    @staticmethod
    def _init_method(method_name, **kwargs):
        return HandEyeCalibrationBase.create(method_name, **kwargs)

    @property
    def _min_samples(self) -> int:
        return 2

    def _get_sample_errors(self, a, b, result):
        calib = result.calib
        if hasattr(result, 'scale'):
            b = b.scaleTranslation(result.scale)
        cycle = (a * calib) / (calib * b)
        return {'rot': cycle.rotationNorm(), 'trans': cycle.translationNorm()}

    def _is_inlier(self, a, b, result):
        errors = self._get_sample_errors(a, b, result)
        return errors['rot'] <= self._rot_thresh and errors['trans'] <= self._trans_thresh


class HandEyeRANSACMultiScale(HandEyeCalibrationBase):
    @staticmethod
    def name():
        return 'HandEyeRANSACMultiScale'

    def __init__(self, *args, **kwargs):
        super(HandEyeCalibrationBase, self).__init__()
        self._ransac = HandEyeRANSAC(*args, **kwargs)
        self._transforms_list_a = None
        self._transforms_list_b = None

    def configure(self, **kwargs):
        self._ransac.configure(**kwargs)

    def set_transforms(self, transforms_a: List[m3d.TransformContainer], transforms_b: List[m3d.TransformContainer],
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self._transforms_list_a = transforms_a
        self._transforms_list_b = transforms_b

    def set_transform_data(self, data) -> None:
        raise NotImplementedError("Multiple transforms are not supported by RANSAC")

    def _calibrate(self, **kwargs):
        # calibrate all datasets separately
        transforms_list_a_inliers = []
        transforms_list_b_inliers = []
        inlier_count = 0
        run_time = 0.0
        for transforms_a, transforms_b in zip(self._transforms_list_a, self._transforms_list_b):
            # calibrate
            self._ransac.set_transforms(transforms_a, transforms_b)
            results = self._ransac.calibrate(**kwargs)
            if results.run_time is not None:
                run_time += results.run_time

            # get inliers
            if 'inliers' in results.aux_data:
                inliers = results.aux_data['inliers']
            else:
                inliers = []
            inlier_count += len(inliers)
            transforms_list_a_inliers.append(container_subset(transforms_a, inliers))
            transforms_list_b_inliers.append(container_subset(transforms_b, inliers))

        # check inliers
        if inlier_count == 0:
            results = CalibrationResultScaled()
            results.message = "no inliers found in RANSAC"
            return results

        # calibrate simultaneously with all inliers
        self._ransac._method.set_transforms(transforms_list_a_inliers, transforms_list_b_inliers)
        results = self._ransac._method.calibrate(**kwargs)
        results.run_time += run_time
        return results


class _HERWRANSACBase(BaseRANSACWeighted, HERWCalibrationBase):
    def __init__(self, *args, **kwargs):
        super(BaseRANSACWeighted, self).__init__()
        super(HERWCalibrationBase, self).__init__()
        self._initialize(*args, **kwargs)
        assert self._rot_thresh is not None and self._trans_thresh is not None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self._data = (transforms_a, transforms_b)

    def set_transform_data(self, data) -> None:
        raise NotImplementedError("Multiple transforms are not supported by RANSAC")

    @staticmethod
    def _init_method(method_name, **kwargs):
        return HERWCalibrationBase.create(method_name, **kwargs)

    @property
    def _min_samples(self) -> int:
        return 3

    def _get_sample_errors(self, a, b, result):
        calib = result.calib
        cycle = (a * calib.x) / (calib.y * b)
        return {'rot': cycle.rotationNorm(), 'trans': cycle.translationNorm()}

    def _is_inlier(self, a, b, result):
        errors = self._get_sample_errors(a, b, result)
        return errors['rot'] <= self._rot_thresh and errors['trans'] <= self._trans_thresh


class HERWRANSAC(HERWCalibrationBase):
    @staticmethod
    def name():
        return 'HERWRANSAC'

    def __init__(self, *args, **kwargs):
        super(HERWCalibrationBase, self).__init__()
        self._method = _HERWRANSACBase(*args, **kwargs)
        self._t_norms = None

    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        if weights is not None:
            raise NotImplementedError("Weights are currently not supported by RANSAC")
        self.set_transform_data([HERWData(frame_x='', frame_y='',
                                          transforms_a=transforms_a, transforms_b=transforms_b)])
        self._is_multi = False

    def set_transform_data(self, data) -> None:
        # TODO(horn): accumulate data of same x,y pair
        self._data = data
        self._is_multi = True

    def configure(self, **kwargs):
        if 't_norms' in kwargs:
            self._t_norms = kwargs['t_norms']
            del kwargs['t_norms']
        self._method.configure(**kwargs)

    def _calibrate(self, **kwargs):
        # calibrate data individually
        data_results = []
        for herw_data in self._data:
            # set transforms
            self._method.set_transforms(herw_data.transforms_a, herw_data.transforms_b)

            # configure translation norm
            if isinstance(self._t_norms, list):
                self._method.configure(t_norms=self._t_norms)
            elif isinstance(self._t_norms, dict):
                t_norms = [
                    self._t_norms[herw_data.frame_x] if herw_data.frame_x in self._t_norms else None,
                    self._t_norms[herw_data.frame_y] if herw_data.frame_y in self._t_norms else None
                ]
                self._method.configure(t_norms=t_norms)
            elif self._t_norms is not None:
                raise RuntimeError(f"Invalid input for t_norms: {self._t_norms}")

            # calibrate
            result = self._method.calibrate(**kwargs)
            if not result.success:
                return type(result)()
            data_results.append(result)

        # check for calibration with all data
        if len(data_results) == 1:
            result = data_results[0]

            # adjust for multi calibration
            if self._is_multi:
                result.calib.x = {self._data[0].frame_x: result.calib.x}
                result.calib.y = {self._data[0].frame_y: result.calib.y}

        else:
            # select all inliers
            data_inliers = []
            inlier_indices = []
            for herw_idx, (herw_data, results) in enumerate(zip(self._data, data_results)):
                inliers = results.aux_data['inliers']
                data_inliers.append(
                    HERWData(frame_x=herw_data.frame_x, frame_y=herw_data.frame_y,
                             transforms_a=container_subset(herw_data.transforms_a, inliers),
                             transforms_b=container_subset(herw_data.transforms_b, inliers))
                )
                inlier_indices.extend([(herw_idx, inlier_idx) for inlier_idx in inliers])

            # calibrate with all data
            self._method._method.set_transform_data(data_inliers)
            if self._t_norms is not None:
                self._method._method.configure(t_norms=self._t_norms)
            result = self._method._method.calibrate(**kwargs)
            result.run_time += np.sum([r.run_time for r in data_results])

            # adjust for multi calibration
            if not self._is_multi:
                result.calib.x = list(result.calib.x.values())[0]
                result.calib.y = list(result.calib.y.values())[0]

            # aux data
            result.aux_data['inliers'] = inlier_indices

        return result


class RANSACMethod(Enum):
    POINT2POINT = Point2PointRANSAC.name()
    FRAME2FRAME = Frame2FrameRANSAC.name()
    HAND_EYE = HandEyeRANSAC.name()
    HAND_EYE_MULTISCALE = HandEyeRANSACMultiScale.name()
    HERW = HERWRANSAC.name()

    @classmethod
    def from_str(cls, s: str):
        return cls[s.upper()]
