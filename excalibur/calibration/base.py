from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import motion3d as m3d

from excalibur.utils import factory
from excalibur.utils.logging import Message
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
    msgs: List[Message] = field(default_factory=list)
    run_time: Optional[float] = None
    aux_data: dict = field(default_factory=dict)

    def get_messages(self):
        return '. '.join([m.text for m in self.msgs])


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


@dataclass
class MultiTransformPair:
    x: Dict[Union[int, str], m3d.TransformInterface]
    y: Dict[Union[int, str], m3d.TransformInterface]


@dataclass
class PairCalibrationResult(_CalibrationResultBase):
    calib: Optional[Union[TransformPair, MultiTransformPair]] = None
