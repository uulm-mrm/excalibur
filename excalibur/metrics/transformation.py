from dataclasses import dataclass
from typing import Optional

import motion3d as m3d
import numpy as np


@dataclass
class TransformationError:
    rotation: float
    translation: float
    euler_angles: Optional[np.ndarray]
    translation_vec: Optional[np.ndarray]


def _rotation_matrix_angle(rotmat: np.ndarray) -> float:
    rotmat_trace = np.trace(rotmat)
    if rotmat_trace >= 3.0:
        return 0.0
    return np.arccos((np.trace(rotmat) - 1.0) / 2.0)


def transformation_error(prediction: m3d.TransformInterface, ground_truth: m3d.TransformInterface,
                         vectors: bool = False) -> TransformationError:
    # difference
    difference = prediction.inverse() * ground_truth

    # component vectors
    if vectors:
        # convert to euler
        prediction = prediction.asType(m3d.TransformType.kEuler).changeAxes_(m3d.EulerAxes.kSXYZ)
        ground_truth = ground_truth.asType(m3d.TransformType.kEuler).changeAxes_(m3d.EulerAxes.kSXYZ)

        # difference
        euler_angles_diff = prediction.getAngles() - ground_truth.getAngles()
        translation_vec_diff = prediction.getTranslation() - ground_truth.getTranslation()
    else:
        euler_angles_diff = None
        translation_vec_diff = None

    # combine
    return TransformationError(
        rotation=difference.rotationNorm(),
        translation=difference.translationNorm(),
        euler_angles=euler_angles_diff,
        translation_vec=translation_vec_diff,
    )
