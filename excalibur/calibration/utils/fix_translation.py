from dataclasses import dataclass
from typing import List, Optional

import motion3d as m3d
import numpy as np

from ..base import CalibrationResult, PairCalibrationResult
from excalibur.fitting.plane import fit_plane


def _estimate_up_vector(transforms, up_approx=None, use_pos=False):
    if use_pos:
        positions = np.array([t.asType(m3d.TransformType.kQuaternion).getTranslation() for t in transforms])
        plane = fit_plane(positions)
        up_vector = plane.normal
    else:
        z_axes = np.array([t.asType(m3d.TransformType.kMatrix).getRotationMatrix()[:, 2] for t in transforms])
        up_vector = np.sum(z_axes, axis=0)
        up_vector /= np.linalg.norm(up_vector)

    # compare to approx up vector
    if up_approx is not None:
        sprod = up_vector.T @ np.array(up_approx)
        if sprod < -sprod:
            up_vector *= -1.0

    return up_vector


def fix_translation_hand_eye(result: CalibrationResult, transforms_a: m3d.TransformContainer, up_approx: np.ndarray) \
        -> CalibrationResult:
    # estimate up vector
    up_vector = _estimate_up_vector(transforms_a, up_approx=up_approx)

    # check z displacement
    calib_quat = result.calib.asType(m3d.TransformType.kQuaternion)
    gamma = np.dot(calib_quat.getTranslation(), up_vector)

    if gamma < 0.0:
        x_shift_transform = m3d.QuaternionTransform(-2 * gamma * up_vector, [1, 0, 0, 0])
        result.calib = x_shift_transform * result.calib

    return result


def fix_translation_herw_single(
        result: PairCalibrationResult, transforms_a: m3d.TransformContainer, up_approx: np.ndarray,
        frames_x: Optional[List] = None, frames_y: Optional[List] = None) -> PairCalibrationResult:
    # check input
    if isinstance(result.calib.x, m3d.TransformInterface) and isinstance(result.calib.y, m3d.TransformInterface):
        main_calib_x = result.calib.x
        frames_x = None
        frames_y = None
    elif frames_x is not None and frames_y is not None:
        main_calib_x = result.calib.x[frames_x[0]]
    else:
        raise RuntimeError("X and Y frames are required for multiple transformations")

    # estimate up vector in vehicle coordinates
    up_vehicle = _estimate_up_vector(transforms_a.inverse(), up_approx=up_approx)

    # check z displacement
    main_calib_x_quat = main_calib_x.asType(m3d.TransformType.kQuaternion)
    gamma = np.dot(main_calib_x_quat.getTranslation(), up_vehicle)

    if gamma < 0.0:
        # adjust x
        x_shift_transform = m3d.QuaternionTransform(-2 * gamma * up_vehicle, [1, 0, 0, 0])
        if frames_x is None:
            result.calib.x = x_shift_transform * result.calib.x
        else:
            for frame in frames_x:
                result.calib.x[frame] = x_shift_transform * result.calib.x[frame]

        # up vector in world coordinates
        up_world = _estimate_up_vector(transforms_a, up_approx=up_approx)

        # adjust y
        y_shift_transform = m3d.QuaternionTransform(-2 * gamma * up_world, [1, 0, 0, 0])
        if frames_y is None:
            result.calib.y = y_shift_transform * result.calib.y
        else:
            for frame in frames_y:
                result.calib.y[frame] = y_shift_transform * result.calib.y[frame]

    return result


@dataclass
class TransformationGroup:
    frames_x: List
    frames_y: List
    transforms_a: m3d.TransformContainer


def _find_transformation_groups(transform_data: List):
    groups = []
    for data in transform_data:
        # check if any transform already exists in group
        found_group = False
        for group in groups:
            if data.frame_x in group.frames_x or data.frame_y in group.frames_y:
                if data.frame_x not in group.frames_x:
                    group.frames_x.append(data.frame_x)
                if data.frame_y not in group.frames_y:
                    group.frames_y.append(data.frame_y)
                group.transforms_a.extend(data.transforms_a)
                found_group = True
                break

        # add new group if no existing group was found
        if not found_group:
            groups.append(TransformationGroup(
                frames_x=[data.frame_x],
                frames_y=[data.frame_y],
                transforms_a=data.transforms_a.copy(),
            ))
    return groups


def fix_translation_herw(result: PairCalibrationResult, transform_data: List, up_approx: np.ndarray) \
        -> PairCalibrationResult:
    # search for transformation groups
    groups = _find_transformation_groups(transform_data)

    # check single transform
    if len(groups) == 1:
        if len(groups[0].frames_x) == 1 and groups[0].frames_x[0] == '' and \
                isinstance(result.calib.x, dict) and len(result.calib.x) == 1:
            groups[0].frames_x = list(result.calib.x.keys())
        if len(groups[0].frames_y) == 1 and groups[0].frames_y[0] == '' and \
                isinstance(result.calib.y, dict) and len(result.calib.y) == 1:
            groups[0].frames_y = list(result.calib.y.keys())

    # iterate all groups
    for group in groups:
        result = fix_translation_herw_single(result, group.transforms_a, up_approx, group.frames_x, group.frames_y)

    return result
