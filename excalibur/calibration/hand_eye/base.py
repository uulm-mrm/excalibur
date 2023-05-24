from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from ..base import _CalibrationBase


class HandEyeCalibrationBase(_CalibrationBase, metaclass=ABCMeta):
    # Solve the AX = XB problem.

    @abstractmethod
    def set_transforms(self, transforms_a: m3d.TransformContainer, transforms_b: m3d.TransformContainer,
                       weights: Optional[Union[List, np.ndarray]] = None):
        raise NotImplementedError
