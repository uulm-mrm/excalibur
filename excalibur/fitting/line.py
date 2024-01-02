from dataclasses import dataclass

import numpy as np

from excalibur.io.dataclass import DataclassIO


@dataclass
class Line(DataclassIO):
    # line with point and direction
    point: np.ndarray
    direction: np.ndarray
