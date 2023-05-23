from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np

from ..base import _CalibrationBase


class Point2PointCalibrationBase(_CalibrationBase, metaclass=ABCMeta):
    # Solve the a = X(b) problem.

    @abstractmethod
    def set_points(self, points_a: np.ndarray, points_b: np.ndarray,
                   weights: Optional[Union[List, np.ndarray]] = None) -> None:
        raise NotImplementedError
