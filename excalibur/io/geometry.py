from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LineData:
    point1: np.ndarray
    point2: np.ndarray
    length: Optional[float] = None


def load_line_data(filename: str) -> List[LineData]:
    data = np.loadtxt(filename)
    lines = [LineData(point1=data[row, :2].astype(int), point2=data[row, 2:4].astype(int),
                      length=data[row, 4] if data.shape[1] > 4 else None)
             for row in range(data.shape[0])]
    return lines
