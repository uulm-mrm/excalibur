from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from .base import HERWCalibrationBase, HERWData
from ..base import PairCalibrationResult, TransformPair, MultiTransformPair
from excalibur.calibration.hand_eye import HandEyeCalibrationBase
from excalibur.calibration.frame2frame import Frame2FrameCalibrationBase


class SeparableHERWCalibration(HERWCalibrationBase):
    """Hand-eye robot-world calibration separated into hand-eye calibration and subsequent pose set registration."""

    @staticmethod
    def name():
        return 'SeparableHERWCalibration'

    def __init__(self, hand_eye_name, frame2frame_name, hand_eye_kwargs=None, frame2frame_kwargs=None):
        super().__init__()
        self._hand_eye_name = hand_eye_name
        self._frame2frame_name = frame2frame_name
        self._hand_eye_init_kwargs = hand_eye_kwargs if hand_eye_kwargs is not None else {}
        self._frame2frame_init_kwargs = frame2frame_kwargs if frame2frame_kwargs is not None else {}
        self._transform_data = None
        self._is_multi = False

    @property
    def transform_data(self):
        return self._transform_data

    @property
    def is_multi(self):
        return self._is_multi

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

        # split calibration arguments
        hand_eye_kwargs = {}
        if 'hand_eye' in kwargs:
            hand_eye_kwargs = kwargs['hand_eye']
            del kwargs['hand_eye']

        frame2frame_kwargs = {}
        if 'frame2frame' in kwargs:
            frame2frame_kwargs = kwargs['frame2frame']
            del kwargs['frame2frame']

        # initialize output
        result = PairCalibrationResult()

        # accumulate data for hand-eye calibration
        hand_eye_data = {}
        for data in self._transform_data:
            motions_a = data.transforms_a.removeStamps().asMotions()
            motions_b = data.transforms_b.removeStamps().asMotions()
            if data.frame_x in hand_eye_data:
                hand_eye_data[data.frame_x][0].extend(motions_a)
                hand_eye_data[data.frame_x][1].extend(motions_b)
            else:
                hand_eye_data[data.frame_x] = (motions_a, motions_b)

        # perform hand-eye calibration
        hand_eye_results = {}
        for frame, (motions_a, motions_b) in hand_eye_data.items():
            # initialize hand-eye calibration
            hand_eye = HandEyeCalibrationBase.create(self._hand_eye_name, **self._hand_eye_init_kwargs)
            hand_eye.configure(**hand_eye_kwargs)

            # get calibration arguments
            hand_eye_calib_kwargs = {}
            if 't_norms' in kwargs and kwargs['t_norms'] is not None and frame in kwargs['t_norms']:
                hand_eye_calib_kwargs['t_norm'] = kwargs['t_norms'][frame]
                if 'up_approx' in kwargs:
                    hand_eye_calib_kwargs['up_approx'] = kwargs['up_approx']

            # set transforms, calibrate, and store results
            hand_eye.set_transforms(motions_a, motions_b)
            hand_eye_result = hand_eye.calibrate(**hand_eye_calib_kwargs)
            if not hand_eye_result.success:
                return result
            hand_eye_results[frame] = hand_eye_result

        # accumulate data for frame2frame calibration
        frame2frame_data = {}
        for data in self._transform_data:
            poses_a = data.transforms_a
            poses_b = data.transforms_b.applyPost(hand_eye_results[data.frame_x].calib.inverse())
            if data.frame_y in frame2frame_data:
                frame2frame_data[data.frame_y][0].extend(poses_a)
                frame2frame_data[data.frame_y][1].extend(poses_b)
            else:
                frame2frame_data[data.frame_y] = (poses_a, poses_b)

        # perform frame2frame calibration
        frame2frame_results = {}
        for frame, (poses_a, poses_b) in frame2frame_data.items():
            # initialize frame2frame registration
            frame2frame = Frame2FrameCalibrationBase.create(self._frame2frame_name, **self._frame2frame_init_kwargs)
            frame2frame.configure(**frame2frame_kwargs)

            # set transforms, calibrate, and store results
            frame2frame.set_transforms(poses_a, poses_b)
            frame2frame_result = frame2frame.calibrate()
            if not frame2frame_result.success:
                return result
            frame2frame_results[frame] = frame2frame_result

        # create final result
        result.success = True
        result.run_time = np.sum([r.run_time for r in hand_eye_results.values()]) + \
                          np.sum([r.run_time for r in frame2frame_results.values()])
        if self._is_multi:
            result.calib = MultiTransformPair(
                x={frame: r.calib for frame, r in hand_eye_results.items()},
                y={frame: r.calib for frame, r in frame2frame_results.items()},
            )
        else:
            assert len(hand_eye_results) == len(frame2frame_results) == 1
            result.calib = TransformPair(
                x=list(hand_eye_results.values())[0].calib,
                y=list(frame2frame_results.values())[0].calib,
            )
        result.aux_data = {
            'hand_eye_results': hand_eye_results,
            'frame2frame_results': frame2frame_results,
        }
        return result
