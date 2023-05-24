import copy
from dataclasses import dataclass
import json
from typing import List, Optional

import motion3d as m3d
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


@dataclass
class Plane:
    normal: np.ndarray
    distance: float

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        normal = np.array(data['normal'])
        distance = data['distance']
        return cls(normal=normal, distance=distance)

    def get_transform(self):
        target_axis = np.array([0.0, 0.0, 1.0])
        n = np.cross(self.normal, target_axis)
        phi = np.arccos(np.dot(self.normal, target_axis))
        translation = np.array([0, 0, self.distance])
        return m3d.AxisAngleTransform(translation, phi, n / np.linalg.norm(n)).normalized_()

    def transform(self, t: m3d.TransformInterface):
        plane_point_new = t.transformPoint(self.normal * self.distance)
        plane_normal_new = t.asType(m3d.TransformType.kMatrix).getRotationMatrix() @ self.normal
        plane_distance_new = np.dot(plane_point_new, plane_normal_new)
        return Plane(plane_normal_new, plane_distance_new)
