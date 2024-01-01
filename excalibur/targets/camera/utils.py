from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import cv2
import motion3d as m3d
import numpy as np


@dataclass
class MarkerDetection:
    corners: np.ndarray
    identifier: Optional[int] = None
    pose: Optional[m3d.TransformInterface] = None
    pose_options: Optional[List[m3d.TransformInterface]] = None


def fix_aruco_params(params: Union[Dict, cv2.aruco.DetectorParameters]) -> cv2.aruco.DetectorParameters:
    if isinstance(params, cv2.aruco.DetectorParameters):
        return params
    else:
        new_params = cv2.aruco.DetectorParameters()
        for k, v in params.items():
            new_params.__setattr__(k, v)
        return new_params
