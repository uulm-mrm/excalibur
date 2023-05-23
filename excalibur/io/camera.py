from dataclasses import dataclass, field

import numpy as np

from .utils import load_yaml, save_yaml


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @classmethod
    def load(cls, filename):
        data = load_yaml(filename)
        if 'camera_matrix' in data:
            data['camera_matrix'] = np.array(data['camera_matrix'])
        if 'dist_coeffs' in data:
            data['dist_coeffs'] = np.array(data['dist_coeffs'])
        return cls(**data)

    def save(self, filename):
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
        }
        save_yaml(filename, data)
