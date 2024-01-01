from enum import auto, Enum
import random

import motion3d as m3d
import numpy as np

from excalibur.fitting.line import Line
from excalibur.fitting.plane import Plane


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


def get_motion_data(rotation_norm=1.0, translation_norm=1.0, planar=False):
    if planar:
        motion_a = m3d.TransformContainer([
            m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [0, 0, 1]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), -rotation_norm, [0, 0, 1]),
            m3d.AxisAngleTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, [0, 0, 1])
        ], has_poses=False)
    else:
        motion_a = m3d.TransformContainer([
            m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [1, 0, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), rotation_norm, [0, 1, 0]),
            m3d.AxisAngleTransform(translation_norm * np.array([0, 0, 1]), rotation_norm, [0, 0, 1])
        ], has_poses=False)
    calib = m3d.EulerTransform(translation_norm * np.array([1, 0, 1]), rotation_norm, rotation_norm, 0.0)
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


def get_herw_pose_data(rotation_norm=1.0, translation_norm=1.0, planar=False,
                       n_random=None, random_type=RandomType.FIXED):
    if n_random is not None:
        poses_a = m3d.TransformContainer([
            get_random_transform(random_type, rotation_norm, translation_norm)
            for _ in range(n_random)
        ], has_poses=True)
        calib_x = get_random_transform(random_type, rotation_norm, translation_norm)
        calib_y = get_random_transform(random_type, rotation_norm, translation_norm)

    else:
        if planar:
            poses_a = m3d.TransformContainer([
                m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [0, 0, 1]),
                m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), -rotation_norm, [0, 0, 1]),
                m3d.AxisAngleTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, [0, 0, 1])
            ], has_poses=True)
        else:
            poses_a = m3d.TransformContainer([
                m3d.AxisAngleTransform(translation_norm * np.array([1, 0, 0]), rotation_norm, [1, 0, 0]),
                m3d.AxisAngleTransform(translation_norm * np.array([0, 1, 0]), rotation_norm, [0, 1, 0]),
                m3d.AxisAngleTransform(translation_norm * np.array([0, 0, 1]), rotation_norm, [0, 0, 1])
            ], has_poses=True)
        calib_x = m3d.EulerTransform(translation_norm * np.array([1, 0, 1]), rotation_norm, rotation_norm, 0.0)
        calib_y = m3d.EulerTransform(translation_norm * np.array([1, 1, 0]), rotation_norm, -rotation_norm, 0.0)

    poses_b = poses_a.apply(calib_y.inverse(), calib_x)

    return poses_a, poses_b, calib_x, calib_y


def get_target_points(poses):
    return np.array([p.asType(m3d.TransformType.kAxisAngle).getTranslation() for p in poses]).T


def get_point2line_data(line_through_origin, **kwargs):
    # create data
    poses_a, poses_b, calib = get_target_data(**kwargs)
    points_a = get_target_points(poses_a)
    points_b = get_target_points(poses_b)

    # create lines
    if line_through_origin:
        line_points_b = np.zeros((3, points_b.shape[1]))
        line_dirs_b = points_b / np.linalg.norm(points_b, axis=0)
    else:
        line_points_b = -points_b + 1.0
        line_dirs_b = points_b - line_points_b
        line_dirs_b /= np.linalg.norm(line_dirs_b, axis=0)
    lines_b = [Line(point=line_points_b[:, i], direction=line_dirs_b[:, i])
               for i in range(line_points_b.shape[1])]

    return points_a, lines_b, calib


def get_point2plane_data(**kwargs):
    # create data
    poses_a, poses_b, calib = get_target_data(**kwargs)

    # create planes (one plane for each axis of each pose in b)
    points_a_list = []
    points_b_list = []
    plane_normals_b_list = []
    for pa, pb in zip(poses_a, poses_b):
        pa = pa.asType(m3d.TransformType.kMatrix)
        pb = pb.asType(m3d.TransformType.kMatrix)
        point_a = pa.getTranslation()
        point_b = pb.getTranslation()
        rotmat_b = pb.getRotationMatrix()
        points_a_list.extend([point_a, point_a, point_a])
        points_b_list.extend([point_b, point_b, point_b])
        plane_normals_b_list.extend([rotmat_b[:, 0], rotmat_b[:, 1], rotmat_b[:, 2]])
    points_a = np.array(points_a_list).T
    points_b = np.array(points_b_list).T
    plane_normals_b = np.array(plane_normals_b_list).T

    plane_distances_b = np.sum(plane_normals_b * points_b, axis=0)
    planes_b = [Plane(normal=plane_normals_b[:, i], distance=plane_distances_b[i])
                for i in range(len(plane_distances_b))]

    return points_a, planes_b, calib


def add_stamps_to_transforms(transforms, dt=0.1):
    stamps = [m3d.Time.FromSec(dt * s) for s in range(len(transforms))]
    return transforms.addStamps(stamps)
