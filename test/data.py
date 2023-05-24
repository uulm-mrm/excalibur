from enum import auto, Enum
import random

import motion3d as m3d
import numpy as np


class RandomType(Enum):
    FIXED = auto()
    UNIFORM = auto()
    NORMAL = auto()


def get_random_sign():
    return 1 if random.random() < 0.5 else -1


def get_random_value(random_type, param):
    if random_type == RandomType.FIXED:
        return param
    elif random_type == RandomType.UNIFORM:
        return np.random.uniform(low=-param, high=param)
    elif random_type == RandomType.NORMAL:
        return np.random.normal()


def get_uniform_vectord(dim, norm=1.0):
    vec = np.random.normal(loc=0.0, scale=1.0, size=dim)
    vec /= np.linalg.norm(vec)
    return vec * norm


def get_random_transform(random_type, rotation_param, translation_param):
    # rotation
    rotation_norm = get_random_value(random_type, rotation_param)
    rotation_axis = get_uniform_vectord(3)

    # translation
    translation_norm = get_random_value(random_type, translation_param)
    translation = get_uniform_vectord(3, norm=translation_norm)

    return m3d.AxisAngleTransform(translation, rotation_norm, rotation_axis)


def get_motion_data(rotation_norm=1.0, translation_norm=1.0):
    motion_a = m3d.TransformContainer([
        m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [1, 0, 0]),
        m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), rotation_norm, [0, 1, 0]),
        m3d.AxisAngleTransform(translation_norm * np.array([0, 0, 1]), rotation_norm, [0, 0, 1])
    ], has_poses=False)
    calib = m3d.EulerTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, rotation_norm, 0.0)
    motion_b = motion_a.changeFrame(calib)
    return motion_a, motion_b, calib


def get_target_data(rotation_norm=1.0, translation_norm=1.0, n_random=None, random_type=RandomType.FIXED):
    if n_random is not None:
        poses_a = m3d.TransformContainer([
            get_random_transform(random_type, rotation_norm, translation_norm)
            for _ in range(n_random)
        ], has_poses=True)
        calib = get_random_transform(random_type, rotation_norm, translation_norm)

    else:
        poses_a = m3d.TransformContainer([
            m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [1, 0, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), rotation_norm, [0, 1, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 0, 1]), rotation_norm, [0, 0, 1]),
            m3d.AxisAngleTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, [1, 0, 0])
        ], has_poses=True)
        calib = m3d.EulerTransform(translation_norm * np.array([1, 0, 1]), rotation_norm, rotation_norm, 0.0)

    poses_b = poses_a.applyPre(calib.inverse())

    return poses_a, poses_b, calib


def get_hew_pose_data(rotation_norm=1.0, translation_norm=1.0, n_random=None, random_type=RandomType.FIXED):
    if n_random is not None:
        poses_a = m3d.TransformContainer([
            get_random_transform(random_type, rotation_norm, translation_norm)
            for _ in range(n_random)
        ], has_poses=True)
        calib_x = get_random_transform(random_type, rotation_norm, translation_norm)
        calib_y = get_random_transform(random_type, rotation_norm, translation_norm)

    else:
        poses_a = m3d.TransformContainer([
            m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [1, 0, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), rotation_norm, [0, 1, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 0, 1]), rotation_norm, [0, 0, 1])
        ], has_poses=True)
        calib_x = m3d.EulerTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, rotation_norm, 0.0)
        calib_y = m3d.EulerTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, rotation_norm, 0.0)

    poses_b = poses_a.apply(calib_y.inverse(), calib_x)

    return poses_a, poses_b, calib_x, calib_y


def get_target_points(poses):
    return np.array([p.asType(m3d.TransformType.kAxisAngle).getTranslation() for p in poses]).T
