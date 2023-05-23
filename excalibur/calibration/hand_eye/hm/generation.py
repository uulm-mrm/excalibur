from dataclasses import dataclass
from typing import List

import motion3d as m3d
import numpy as np

from excalibur.optimization.qcqp import generate_quadratic_cost_matrix


def gen_linear_andreff(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_b = transforms_b.asType(m3d.TransformType.kMatrix)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate motions
    Alist = list()
    blist = list()

    for Va, Vb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Va.getRotationMatrix()
        ta = Va.getTranslation()
        Rb = Vb.getRotationMatrix()
        tb = Vb.getTranslation()

        # costs
        A = np.vstack([
            np.hstack([         np.eye(9) - np.kron(Ra, Rb), np.zeros((9, 3))]),
            np.hstack([np.kron(np.eye(3), tb.reshape(1, 3)),   np.eye(3) - Ra]),
        ])
        b = np.concatenate([np.zeros((9, 1)), ta.reshape(3, 1)])

        # store
        Alist.append(A)
        blist.append(b)

    # create full matrices
    A = np.vstack(Alist)
    b = np.vstack(blist)

    return A, b


def gen_Mlist(transforms_a, transforms_b, scaled=False, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_b = transforms_b.asType(m3d.TransformType.kMatrix)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate motions
    Mlist_r = list()
    Mlist_t = list()

    for Va, Vb in zip(transforms_a, transforms_b):
        # matrices
        Ra = Va.getRotationMatrix()
        ta = Va.getTranslation()
        Rb = Vb.getRotationMatrix()
        tb = Vb.getTranslation()

        # rotation and translation cost
        M_r = np.kron(Ra.T, np.eye(3)) - np.kron(np.eye(3), Rb)
        if scaled:
            # x = [t, alpha, r]
            M_t = np.hstack([np.eye(3) - Rb, -tb.reshape(3, 1), np.kron(ta.reshape(1, 3), np.eye(3))])
        else:
            # x = [t, r, y]
            M_t = np.hstack([np.eye(3) - Rb, np.kron(ta.reshape(1, 3), np.eye(3)), -tb.reshape(3, 1)])

        # store results
        Mlist_r.append(M_r)
        Mlist_t.append(M_t)

    return Mlist_r, Mlist_t


def gen_Q(Mlist_r, Mlist_t, scaled=False, weights_r=None, weights_t=None, normalize=False):
    Q = generate_quadratic_cost_matrix(Mlist_t, weights=weights_t, normalize=normalize)
    Q_r = generate_quadratic_cost_matrix(Mlist_r, weights=weights_r, normalize=normalize)
    if scaled:
        # x = [t, alpha, r]
        Q[4:, 4:] += Q_r
    else:
        # x = [t, r, y]
        Q[3:-1, 3:-1] += Q_r
    return Q


@dataclass
class SchmidtData:
    eye_hm: List[m3d.DualQuaternionTransform]
    eye_quat: List[m3d.Quaternion]
    hand_hm: List[m3d.DualQuaternionTransform]
    hand_quat: List[m3d.Quaternion]
    e_quat_vec: np.ndarray
    h_quat_neg: np.ndarray
    e_rot_diff: np.ndarray
    e_tran_vec: np.ndarray
    h_tran_quat_neg: np.ndarray


def gen_schmidt(transforms_a, transforms_b, normalize=True):
    # prepare motions
    assert transforms_a.size() == transforms_b.size()
    transforms_a_hm = transforms_a.asType(m3d.TransformType.kMatrix)
    transforms_a_quat = transforms_a.asType(m3d.TransformType.kQuaternion)
    transforms_b_hm = transforms_b.asType(m3d.TransformType.kMatrix)
    transforms_b_quat = transforms_b.asType(m3d.TransformType.kQuaternion)

    if normalize:
        transforms_a_hm.normalized_()
        transforms_a_quat.normalized_()
        transforms_b_hm.normalized_()
        transforms_b_quat.normalized_()

    # iterate motions
    eye_hm_list = []
    eye_quat_list = []
    hand_hm_list = []
    hand_quat_list = []
    e_quat_vec_list = []
    h_quat_neg_list = []
    e_rot_diff_list = []
    e_tran_vec_list = []
    h_tran_quat_neg_list = []
    for Va_hm, Va_quat, Vb_hm, Vb_quat in zip(transforms_a_hm, transforms_a_quat, transforms_b_hm, transforms_b_quat):
        eye_hm_list.append(Va_hm)
        eye_quat_list.append(Va_quat)
        hand_hm_list.append(Vb_hm)
        hand_quat_list.append(Vb_quat)
        e_quat_vec_list.append(Va_quat.getQuaternion().toArray())
        h_quat_neg_list.append(Vb_quat.getQuaternion().toNegativeMatrix())
        e_rot_diff_list.append(np.eye(3) - Va_hm.getRotationMatrix())
        e_tran_vec_list.append(Va_hm.getTranslation())
        h_tran_quat_neg_list.append(m3d.Quaternion(0.0, *(Vb_hm.getTranslation())).toNegativeMatrix())

    # create data
    data = SchmidtData(
        eye_hm=eye_hm_list,
        eye_quat=eye_quat_list,
        hand_hm=hand_hm_list,
        hand_quat=hand_quat_list,
        e_quat_vec=np.concatenate(e_quat_vec_list),
        h_quat_neg=np.stack(h_quat_neg_list),
        e_rot_diff=np.stack(e_rot_diff_list),
        e_tran_vec=np.stack(e_tran_vec_list),
        h_tran_quat_neg=np.vstack(h_tran_quat_neg_list),
    )
    return data
