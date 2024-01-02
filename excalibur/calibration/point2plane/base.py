from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np

from excalibur.fitting.plane import Plane

from ..base import _CalibrationBase


class Point2PlaneCalibrationBase(_CalibrationBase, metaclass=ABCMeta):
    # Minimize the distances between points in frame a and planes in frame b.

    @abstractmethod
    def set_data(self, points_a: np.ndarray, planes_b: List[Plane],
                 weights: Optional[Union[List, np.ndarray]] = None) -> None:
        raise NotImplementedError
