from dataclasses import dataclass
import itertools
from typing import List

import motion3d as m3d
import numpy as np

from ..base import HERWCalibrationBase, HERWData, FrameIds
from ...base import PairCalibrationResult


def _get_sign_permutations(n_samples):
    permutations = []
    for signs in itertools.product([1.0, -1.0], repeat=n_samples):
        signs_inv = [-s for s in signs]
        if signs_inv not in permutations:
            permutations.append(list(signs))
    return permutations


def _adjust_sign(M, sign):
    assert M.shape == (8, 16)
    Mcopy = M.copy()
    Mcopy[:, 8:] *= sign
    return Mcopy


def calibrate_herw_sign_sampling(
        method: HERWCalibrationBase, transforms_a, transforms_b, frame_ids=None, weights=None, n_reps=1, n_samples=3):
    # set transforms
    method.set_transforms(transforms_a, transforms_b, weights=weights)
    if frame_ids is not None:
        assert len(frame_ids.x) == 1 and len(frame_ids.y) == 1
        method.set_frame_ids(frame_ids)

    # copy Mlist
    Mlist_base = method.Mlist.copy()

    # initialize sign perturbations
    sign_perturbations = _get_sign_permutations(n_samples)

    # repeat multiple times
    best_result = None
    for _ in range(n_reps):
        # random indices
        sample_indices = np.random.choice(np.arange(len(transforms_a)), n_samples, replace=False)

        # iterate sign permutations
        best_result_sign = None
        for signs in sign_perturbations:
            # select samples Ms and adjust signs
            sample_Mlist = [_adjust_sign(Mlist_base[sidx], sign) for sidx, sign in zip(sample_indices, signs)]

            # calibrate
            method.set_Mlist(sample_Mlist, weights=weights)
            result_sign = method.calibrate()
            if not result_sign.success:
                continue

            # compare costs
            if best_result_sign is None or result_sign.aux_data['cost'] < best_result_sign.aux_data['cost']:
                best_result_sign = result_sign

        # compare to best result
        if best_result_sign is None:
            continue
        if best_result is None or best_result_sign.aux_data['cost'] < best_result.aux_data['cost']:
            best_result = best_result_sign

    # initialize overall result
    result = PairCalibrationResult()

    # check if any calibration had success
    if best_result is None:
        return result

    # create solution vector from calib data
    if frame_ids is None:
        calib_x = best_result.calib.x
        calib_y = best_result.calib.y
    else:
        calib_x = best_result.calib.x[frame_ids.x[0]]
        calib_y = best_result.calib.y[frame_ids.y[0]]
    z = np.concatenate((calib_x.asType(m3d.TransformType.kDualQuaternion).toArray(),
                        calib_y.asType(m3d.TransformType.kDualQuaternion).toArray()))

    # update all signs based on z
    for Midx in range(len(Mlist_base)):
        M1 = Mlist_base[Midx]
        M2 = M1.copy()
        M2[:, :8] *= -1.0
        if z.T @ M2.T @ M2 @ z <= z.T @ M1.T @ M1 @ z:
            Mlist_base[Midx][:, :8] *= -1.0

    # calibrate with updated Mlist
    method.set_Mlist(Mlist_base, weights=weights)
    result = method.calibrate(x0=z)
    return result


@dataclass
class _CalibData:
    transform: m3d.TransformInterface
    n_samples: int


def calibrate_herw_sign_sampling_multi(
        method: HERWCalibrationBase, data: List[HERWData], n_reps: int = 1, n_samples: int = 3):
    # calibrate all subsets separately
    calib_data_x = {}
    calib_data_y = {}
    for d in data:
        # optimize
        frame_ids = FrameIds(x=[d.frame_x], y=[d.frame_y])
        sub_result = calibrate_herw_sign_sampling(method, d.transforms_a, d.transforms_b, frame_ids=frame_ids,
                                                  weights=d.weights, n_reps=n_reps, n_samples=n_samples)
        if not sub_result.success:
            continue

        # store
        if d.frame_x not in calib_data_x or len(d.transforms_a) > calib_data_x[d.frame_x].n_samples:
            calib_data_x[d.frame_x] = _CalibData(transform=sub_result.calib.x[d.frame_x], n_samples=len(d.transforms_a))
        if d.frame_y not in calib_data_y or len(d.transforms_a) > calib_data_y[d.frame_y].n_samples:
            calib_data_y[d.frame_y] = _CalibData(transform=sub_result.calib.y[d.frame_y], n_samples=len(d.transforms_a))

    # add all transformations to method
    method.set_transform_data(data)

    # create solution vector from separate calib data
    z_vecs = []
    for frame_id in method.frame_ids.x:
        if frame_id not in calib_data_x:
            return PairCalibrationResult(success=False)
        z_vecs.append(calib_data_x[frame_id].transform.asType(m3d.TransformType.kDualQuaternion).toArray())
    for frame_id in method.frame_ids.y:
        if frame_id not in calib_data_y:
            return PairCalibrationResult(success=False)
        z_vecs.append(calib_data_y[frame_id].transform.asType(m3d.TransformType.kDualQuaternion).toArray())
    z = np.concatenate(z_vecs)

    # update all signs based on z
    num_frames_x = len(method.frame_ids.x)

    Mlist_new = []
    for M1 in method.Mlist:
        M2 = M1.copy()
        M2[:, :8 * num_frames_x] *= -1.0
        if z.T @ M1.T @ M1 @ z <= z.T @ M2.T @ M2 @ z:
            Mlist_new.append(M1)
        else:
            Mlist_new.append(M2)

    # calibrate with updated Mlist
    method.set_Mlist(Mlist_new)
    result = method.calibrate(x0=z)
    return result
