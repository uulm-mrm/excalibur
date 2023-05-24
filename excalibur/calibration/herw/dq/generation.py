from typing import List

import motion3d as m3d
import numpy as np

from ..base import FrameIds, HERWData
from excalibur.utils.math import canonical_vector


def gen_Mlist(transforms_a, transforms_b, normalize=True, use_C=False, use_D=False):
    data = [HERWData(frame_x=0, frame_y=0, transforms_a=transforms_a, transforms_b=transforms_b)]
    Mlist, _, _ = gen_Mlist_multi(data, normalize=normalize, use_C=use_C, use_D=use_D)
    return Mlist


def gen_Mlist_multi(data: List[HERWData], normalize=True, use_C=False, use_D=False, use_all=False):
    # check input
    assert isinstance(data, list)

    # check frames
    frames_x = list(set([d.frame_x for d in data]))
    frames_y = list(set([d.frame_y for d in data]))
    frame_idx_x = {frame: frame_id for frame_id, frame in enumerate(frames_x)}
    frame_idx_y = {frame: frame_id for frame_id, frame in enumerate(frames_y)}
    x_count = len(frames_x)
    y_count = len(frames_y)
    frame_ids = FrameIds(x=frames_x, y=frames_y)

    # prepare output
    Mlist = []
    weights = None

    # iterate
    for d in data:
        # prepare motions
        assert d.transforms_a.size() == d.transforms_b.size()
        transforms_a = d.transforms_a.asType(m3d.TransformType.kDualQuaternion)
        transforms_b = d.transforms_b.asType(m3d.TransformType.kDualQuaternion)

        if normalize:
            transforms_a.normalized_()
            transforms_b.normalized_()

        # prepare canonical vectors
        can_vec_x = canonical_vector(x_count, frame_idx_x[d.frame_x]).reshape(1, x_count)
        can_vec_y = canonical_vector(y_count, frame_idx_y[d.frame_y]).reshape(1, y_count)

        # iterate motions
        for Ta, Tb in zip(transforms_a, transforms_b):
            # matrices
            A = Ta.getDualQuaternion().toPositiveMatrix()
            Ainv = Ta.getDualQuaternion().inverse().toPositiveMatrix()
            B = Tb.getDualQuaternion().toNegativeMatrix()
            Binv = Tb.getDualQuaternion().inverse().toNegativeMatrix()

            # extra matrices
            C = Ainv @ B
            D = A @ Binv

            # flip check
            if C[0, 0] < 0.0:
                B *= -1.0
                C *= -1.0
                D *= -1.0

            # create and store M
            if use_C:
                M = np.column_stack([
                    np.kron(can_vec_x, np.eye(8)),
                    np.kron(can_vec_y, C),
                ])
                Mlist.append(M)
            elif use_D:
                M = np.column_stack([
                    np.kron(can_vec_x, D),
                    np.kron(can_vec_y, np.eye(8)),
                ])
                Mlist.append(M)
            elif use_all:
                M = np.column_stack([
                    np.kron(can_vec_x, np.eye(8)),
                    np.kron(can_vec_y, C),
                ])
                Mlist.append(M)
                M = np.column_stack([
                    np.kron(can_vec_x, D),
                    np.kron(can_vec_y, np.eye(8)),
                ])
                Mlist.append(M)
                M = np.column_stack([
                    np.kron(can_vec_x, A),
                    np.kron(can_vec_y, B),
                ])
                Mlist.append(M)
            else:
                M = np.column_stack([
                    np.kron(can_vec_x, A),
                    np.kron(can_vec_y, B),
                ])
                Mlist.append(M)

        # weights
        if d.weights is not None:
            if weights is None:
                weights = []
            weights.extend(d.weights)

    # check weights
    assert weights is None or len(Mlist) == len(weights)
    return Mlist, frame_ids, weights
