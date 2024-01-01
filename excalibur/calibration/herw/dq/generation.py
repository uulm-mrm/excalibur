from enum import auto, Enum
from typing import List

import motion3d as m3d
import numpy as np

from ..base import FrameIds, HERWData
from excalibur.utils.math import canonical_vector


class QCQPDQCostFun(Enum):
    AX_YB = auto()
    X_AiYB = auto()
    AXBi_Y = auto()
    ALL = auto()


DEFAULT_COST_FUN = QCQPDQCostFun.X_AiYB


def gen_Mlist(transforms_a, transforms_b, normalize=True, cost_fun=DEFAULT_COST_FUN):
    data = [HERWData(frame_x=0, frame_y=0, transforms_a=transforms_a, transforms_b=transforms_b)]
    Mlist, _, _ = gen_Mlist_multi(data, normalize=normalize, cost_fun=cost_fun)
    return Mlist


def gen_Mlist_multi(data: List[HERWData], normalize=True, cost_fun=DEFAULT_COST_FUN):
    # check input
    assert isinstance(data, list)

    # handle cost functions
    if cost_fun == QCQPDQCostFun.ALL:
        Mlist_results = [gen_Mlist_multi(data, normalize, cost_fun=cf)
                         for cf in [QCQPDQCostFun.AX_YB, QCQPDQCostFun.X_AiYB, QCQPDQCostFun.AXBi_Y]]
        Mlist = [tmp[0] for tmp in Mlist_results]
        frame_ids = Mlist_results[0][1]
        weights = [tmp[2] for tmp in Mlist_results]
        return Mlist, frame_ids, weights

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
        # prepare poses
        assert d.transforms_a.hasPoses() and d.transforms_b.hasPoses()
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

            # create and store M
            if cost_fun == cost_fun.AX_YB:
                M = np.column_stack([
                    np.kron(can_vec_x, A),
                    np.kron(can_vec_y, B),
                ])
            elif cost_fun == cost_fun.X_AiYB:
                C = Ainv @ B
                M = np.column_stack([
                    np.kron(can_vec_x, np.eye(8)),
                    np.kron(can_vec_y, C),
                ])
            elif cost_fun == cost_fun.AXBi_Y:
                D = A @ Binv
                M = np.column_stack([
                    np.kron(can_vec_x, D),
                    np.kron(can_vec_y, np.eye(8)),
                ])
            else:
                raise NotImplementedError(f"Unsupported cost function: {cost_fun}")

            Mlist.append(M)

        # weights
        if d.weights is not None:
            if weights is None:
                weights = []
            weights.extend(d.weights)

    # check weights
    assert weights is None or len(Mlist) == len(weights)
    return Mlist, frame_ids, weights
