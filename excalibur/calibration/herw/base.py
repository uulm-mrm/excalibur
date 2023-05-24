from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from ..base import _CalibrationBase


FrameId = Union[int, str]


@dataclass
class FrameIds:
    x: List[Union[int, str]]
    y: List[Union[int, str]]


@dataclass
class HERWData:
    frame_x: FrameId
    frame_y: FrameId
    transforms_a: m3d.TransformContainer
    transforms_b: m3d.TransformContainer
    weights: Optional[Union[List, np.ndarray]] = None


class HERWCalibrationBase(_CalibrationBase, metaclass=ABCMeta):
    # Solve the AX = YB problem.

    @abstractmethod
    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None) -> None:
        raise NotImplementedError
