from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import motion3d as m3d

from excalibur.utils import factory
from excalibur.utils.parameters import add_default_kwargs


class _CalibrationBase(ABC):
    @classmethod
    def create(cls, name, **kwargs):
        return factory.get_child(cls, name, **kwargs)

    @staticmethod
    def name():
        return None

    @staticmethod
    def _calibrate(self, **kwargs):
        raise NotImplementedError

    def __init__(self):
        self._kwargs = {}

    def configure(self, **kwargs):
        self._kwargs = add_default_kwargs(kwargs, **self._kwargs)

    def calibrate(self, **kwargs):
        kwargs = add_default_kwargs(kwargs, **self._kwargs)
        return self._calibrate(**kwargs)


@dataclass
class _CalibrationResultBase:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    aux_data: dict = field(default_factory=dict)


@dataclass
class CalibrationResult(_CalibrationResultBase):
    calib: Optional[m3d.TransformInterface] = None


@dataclass
class CalibrationResultScaled(CalibrationResult):
    scale: Optional[Union[float, List[float]]] = None


@dataclass
class TransformPair:
    x: m3d.TransformInterface
    y: m3d.TransformInterface

    def get_x(self, frame_id):
        if frame_id == '':
            return self.x
        else:
            raise RuntimeError("Invalid calibration access.")

    def get_y(self, frame_id):
        if frame_id == '':
            return self.y
        else:
            raise RuntimeError("Invalid calibration access.")


@dataclass
class MultiTransformPair:
    x: Dict[Union[int, str], m3d.TransformInterface]
    y: Dict[Union[int, str], m3d.TransformInterface]

    def get_x(self, frame_id):
        if frame_id == '' and (len(self.x) != 1 or frame_id not in self.x):
            raise RuntimeError("Invalid calibration access.")
        return self.x[frame_id]

    def get_y(self, frame_id):
        if frame_id == '' and (len(self.y) != 1 or frame_id not in self.y):
            raise RuntimeError("Invalid calibration access.")
        return self.y[frame_id]


@dataclass
class PairCalibrationResult(_CalibrationResultBase):
    calib: Optional[Union[TransformPair, MultiTransformPair]] = None
