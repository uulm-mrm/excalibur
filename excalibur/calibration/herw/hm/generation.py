from dataclasses import dataclass
from typing import Tuple

import motion3d as m3d
import numpy as np

from ..base import FrameIds
from excalibur.utils.math import canonical_vector


def gen_linear_li(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_b = transforms_b.asType(m3d.TransformType.kMatrix)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate poses
    Alist = list()
    blist = list()
    for Ta, Tb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Ta.getRotationMatrix()
        ta = Ta.getTranslation()
        Rb = Tb.getRotationMatrix()
        tb = Tb.getTranslation()

        # costs
        A = np.vstack([
            np.hstack([np.kron(Ra, np.eye(3)),              np.kron(-np.eye(3), Rb.T), np.zeros((9, 6))]),
            np.hstack([      np.zeros((3, 9)), np.kron(np.eye(3), tb.reshape((1, 3))), -Ra, np.eye(3)]),
        ])
        b = np.vstack([np.zeros((9, 1)), ta.reshape((3, 1))])

        # store
        Alist.append(A)
        blist.append(b)

    # create full matrices
    A = np.vstack(Alist)
    b = np.vstack(blist)

    return A, b


def gen_shah(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_b = transforms_b.asType(m3d.TransformType.kMatrix)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate poses
    T = np.zeros((9, 9))
    Alist = list()
    b_data = list()
    for Ta, Tb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Ta.getRotationMatrix()
        ta = Ta.getTranslation()
        Rb = Tb.getRotationMatrix()
        tb = Tb.getTranslation()

        # costs
        T += np.kron(Rb, Ra)
        A = np.hstack([-Ra, np.eye(3)])
        b = (ta.reshape(3, 1), np.kron(tb.reshape(1, 3), np.eye(3)))

        # store
        Alist.append(A)
        b_data.append(b)

    # create full matrices
    A = np.vstack(Alist)

    return T, A, b_data


@dataclass
class MatrixData:
    A: np.ndarray
    B: np.ndarray
    x_idx: int = 0
    y_idx: int = 0


def gen_matrix_data(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_b = transforms_b.asType(m3d.TransformType.kMatrix)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate poses
    A_list = [t.getMatrix() for t in transforms_a]
    B_list = [t.getMatrix() for t in transforms_b]

    # matrix data
    data = MatrixData(
        A=np.stack(A_list),
        B=np.stack(B_list),
    )
    return data


def gen_matrix_data_multi(transforms_data, normalize=True):
    assert isinstance(transforms_data, list)

    # check x and y count
    frames_x = np.unique([data.frame_x for data in transforms_data])
    frame_idx_x = {frame: frame_id for frame_id, frame in enumerate(frames_x)}

    frames_y = np.unique([data.frame_y for data in transforms_data])
    frame_idx_y = {frame: frame_id for frame_id, frame in enumerate(frames_y)}

    frame_ids = FrameIds(x=frames_x, y=frames_y)

    # iterate
    matrix_data_list = []
    for mfd in transforms_data:
        single_data = gen_matrix_data(mfd.transforms_a, mfd.transforms_b, normalize=normalize)
        single_data.x_idx = frame_idx_x[mfd.frame_x]
        single_data.y_idx = frame_idx_y[mfd.frame_y]
        matrix_data_list.append(single_data)

    return matrix_data_list, frame_ids


@dataclass
class WangData:
    M_rot: np.ndarray
    M_trans: np.ndarray
    t_trans_a: np.ndarray
    t_trans_b: np.ndarray
    swap: bool


def gen_wang(transforms_data, normalize=True, swap=False) -> Tuple[WangData, FrameIds]:
    # check input
    assert isinstance(transforms_data, list)

    # check x and y count
    frames_x = np.unique([data.frame_x for data in transforms_data])
    frame_idx_x = {frame: frame_id for frame_id, frame in enumerate(frames_x)}
    x_count = len(frames_x)

    frames_y = np.unique([data.frame_y for data in transforms_data])
    # frame_idx_y = {frame: frame_id for frame_id, frame in enumerate(frames_y)}
    y_count = len(frames_y)

    if x_count != 1 and y_count != 1:
        raise RuntimeError("Either multiple x or multiple y are allowed")

    frame_ids = FrameIds(x=frames_x, y=frames_y)

    # flip if multiple y
    if y_count > 1:
        transforms_data_inv = [(tid_b, tid_a, tb.inverse(), ta.inverse())
                               for tid_a, tid_b, ta, tb in transforms_data]
        return gen_wang(transforms_data_inv, normalize=normalize, swap=True)

    # prepare output
    M_rot_list = []
    M_trans_list = []
    t_trans_a_list = []
    t_trans_b_list = []

    # iterate
    for mfd in transforms_data:

        # prepare motions
        assert mfd.transforms_a.size() == mfd.transforms_b.size()
        transforms_a = mfd.transforms_a.asType(m3d.TransformType.kMatrix)
        transforms_b = mfd.transforms_b.asType(m3d.TransformType.kMatrix)

        if normalize:
            transforms_a.normalized_()
            transforms_b.normalized_()

        # prepare canonical vector
        can_vec = canonical_vector(x_count, frame_idx_x[mfd.frame_x]).reshape(1, x_count)

        # iterate poses
        for Ta, Tb in zip(transforms_a, transforms_b):
            # matrices
            Ra = Ta.getRotationMatrix()
            ta = Ta.getTranslation()
            Rb = Tb.getRotationMatrix()
            tb = Tb.getTranslation()

            # rotation
            M_rot_list.append(np.hstack([
                -np.eye(9), np.kron(can_vec, np.kron(Rb, Ra))
            ]))

            # translation
            M_trans_list.append(np.hstack([
                np.eye(3), np.kron(can_vec, -Ra)
            ]))
            t_trans_a_list.append(ta.reshape(3, 1))
            t_trans_b_list.append(tb.reshape(3, 1))

    # create full matrices
    M_rot = np.vstack(M_rot_list)
    M_rot = M_rot.T @ M_rot
    M_trans = np.vstack(M_trans_list)
    t_trans_a = np.vstack(t_trans_a_list)
    t_trans_b = np.column_stack(t_trans_b_list)

    return WangData(M_rot, M_trans, t_trans_a, t_trans_b, swap), frame_ids
