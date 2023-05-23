from dataclasses import dataclass, field
from typing import List

import motion3d as m3d
import numpy as np


def _reduce_daniilidis(quat_mat):
    quat_mat = quat_mat[1:, :]
    quat_mat[0, 1] = 0.0
    quat_mat[1, 2] = 0.0
    quat_mat[2, 3] = 0.0
    return quat_mat


def gen_Mlist(transforms_a, transforms_b, daniilidis=False, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kDualQuaternion)
    transforms_b = transforms_b.asType(m3d.TransformType.kDualQuaternion)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate motions
    Mlist = list()
    for Va, Vb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Va.getReal().toPositiveMatrix()
        Da = Va.getDual().toPositiveMatrix()
        Rb = Vb.getReal().toNegativeMatrix()
        Db = Vb.getDual().toNegativeMatrix()

        # daniilidis
        if daniilidis:
            Ra = _reduce_daniilidis(Ra)
            Da = _reduce_daniilidis(Da)
            Rb = _reduce_daniilidis(Rb)
            Db = _reduce_daniilidis(Db)
            height = 3
        else:
            height = 4

        # create M
        M = np.vstack([
            np.hstack([Ra - Rb, np.zeros((height, 4))]),
            np.hstack([Da - Db, Ra - Rb])
        ])

        # store result
        Mlist.append(M)

    return Mlist


def gen_Mlist_scaled(transforms_list_a, transforms_list_b, daniilidis=False, normalize=True):
    # check input
    assert len(transforms_list_a) == len(transforms_list_b)
    scale_count = len(transforms_list_a)

    # iterate
    Mlist = list()
    for scale_index, (transforms_a, transforms_b) in enumerate(zip(transforms_list_a, transforms_list_b)):
        # prepare motions
        assert transforms_a.size() == transforms_b.size()
        transforms_a = transforms_a.asType(m3d.TransformType.kDualQuaternion)
        transforms_b = transforms_b.asType(m3d.TransformType.kDualQuaternion)

        if normalize:
            transforms_a.normalized_()
            transforms_b.normalized_()

        # iterate motions
        for Va, Vb in zip(transforms_a, transforms_b):
            # matrices
            Ra = Va.getReal().toPositiveMatrix()
            Da = Va.getDual().toPositiveMatrix()
            Rb = Vb.getReal().toNegativeMatrix()
            Db = Vb.getDual().toNegativeMatrix()

            # daniilidis
            if daniilidis:
                Ra = _reduce_daniilidis(Ra)
                Da = _reduce_daniilidis(Da)
                Rb = _reduce_daniilidis(Rb)
                Db = _reduce_daniilidis(Db)
                height = 3
            else:
                height = 4

            # scaling
            M = np.vstack([
                np.hstack([Ra - Rb, np.zeros((height, 4 + 4 * scale_count))]),
                np.hstack([Da,
                           Ra - Rb,
                           np.zeros((height, 4 * scale_index)),
                           -Db,
                           np.zeros((height, 4 * (scale_count - scale_index - 1)))])
            ])

            # store result
            Mlist.append(M)

    return Mlist


@dataclass
class SchmidtData:
    eye: List[m3d.DualQuaternionTransform]
    hand: List[m3d.DualQuaternionTransform]
    e_real_pos: np.ndarray
    e_dual_pos: np.ndarray
    h_real_neg: np.ndarray
    h_dual_neg: np.ndarray


def gen_schmidt(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kDualQuaternion)
    transforms_b = transforms_b.asType(m3d.TransformType.kDualQuaternion)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate motions
    eye_list = []
    hand_list = []
    e_real_pos_list = []
    e_dual_pos_list = []
    h_real_neg_list = []
    h_dual_neg_list = []
    for Va, Vb in zip(transforms_a, transforms_b):
        eye_list.append(Va)
        hand_list.append(Vb)
        e_real_pos_list.append(Va.getReal().toPositiveMatrix())
        e_dual_pos_list.append(Va.getDual().toPositiveMatrix())
        h_real_neg_list.append(Vb.getReal().toNegativeMatrix())
        h_dual_neg_list.append(Vb.getDual().toNegativeMatrix())

    # create data
    data = SchmidtData(
        eye=eye_list,
        hand=hand_list,
        e_real_pos=np.vstack(e_real_pos_list),
        e_dual_pos=np.vstack(e_dual_pos_list),
        h_real_neg=np.vstack(h_real_neg_list),
        h_dual_neg=np.vstack(h_dual_neg_list),
    )
    return data


@dataclass
class WeiData:
    V: np.ndarray
    Cs: np.ndarray
    ds: np.ndarray


def gen_wei(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kDualQuaternion)
    transforms_b = transforms_b.asType(m3d.TransformType.kDualQuaternion)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate motions
    V_list = []
    Cs_list = []
    ds_list = []
    for Va, Vb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Va.getReal().toPositiveMatrix()
        Da = Va.getDual().toPositiveMatrix()
        Rb = Vb.getReal().toNegativeMatrix()
        Db = Vb.getDual().toNegativeMatrix()

        # create and store matrices
        V_list.append(Ra - Rb)
        Cs_list.append(np.hstack([Ra - Rb, -Db]))
        ds_list.append(-Da)

    # stack and return
    data = WeiData(
        V=np.vstack(V_list),
        Cs=np.vstack(Cs_list),
        ds=np.vstack(ds_list),
    )
    return data
