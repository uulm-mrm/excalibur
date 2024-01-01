from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np

from excalibur.fitting.line import Line

from ..base import _CalibrationBase


class Point2LineCalibrationBase(_CalibrationBase, metaclass=ABCMeta):
    # Minimize the distances between points in frame a and lines in frame b.

    @abstractmethod
    def set_data(self, points_a: np.ndarray, lines_b: List[Line],
                 weights: Optional[Union[List, np.ndarray]] = None) -> None:
        raise NotImplementedError
