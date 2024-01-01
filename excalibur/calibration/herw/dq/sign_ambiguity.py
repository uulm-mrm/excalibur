from dataclasses import dataclass
import itertools
from typing import List, Optional, Union

import motion3d as m3d
import numpy as np

from ..base import HERWCalibrationBase, HERWData, FrameIds
from ...base import MultiTransformPair, PairCalibrationResult, TransformPair


def _get_sign_permutations(n_samples):
    permutations = []
    for signs in itertools.product([1.0, -1.0], repeat=n_samples):
        signs_inv = [-s for s in signs]
        if signs_inv not in permutations:
            permutations.append(list(signs))
            yield signs


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
    # Mlist is either List[List[np.ndarray]] for QCQPDQCostFun.ALL) or List[np.ndarray] for other cost functions
    if isinstance(method.Mlist[0], list):
        raise RuntimeError("Sign sampling is not supported for QCQPDQCostFun.ALL")
    Mlist_base = method.Mlist.copy()

    # initialize sign perturbations
    sign_perturbations = _get_sign_permutations(n_samples)

    # repeat multiple times
    best_result = None
    run_time = 0.0
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
            run_time += result_sign.run_time
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

    # check if any calibration had success
    if best_result is None:
        return PairCalibrationResult()

    # normalize Mlist
    Mlist_norm, z = normalize_Mlist(Mlist_base, best_result.calib, frame_ids=frame_ids)

    # calibrate with updated Mlist
    method.set_Mlist(Mlist_norm, weights=weights)
    result = method.calibrate(x0=z)
    result.run_time += run_time
    return result


def calibrate_herw_sign_brute_force(
        method: HERWCalibrationBase, transforms_a, transforms_b, frame_ids=None, weights=None):
    # set transforms
    method.set_transforms(transforms_a, transforms_b, weights=weights)
    if frame_ids is not None:
        assert len(frame_ids.x) == 1 and len(frame_ids.y) == 1
        method.set_frame_ids(frame_ids)

    # copy Mlist
    Mlist_base = method.Mlist.copy()

    # initialize sign perturbations
    sign_perturbations = _get_sign_permutations(len(transforms_a))

    # brute force signs
    best_result = None
    run_time = 0.0
    for signs in sign_perturbations:
        # select samples Ms and adjust signs
        sample_Mlist = [_adjust_sign(M, sign) for M, sign in zip(Mlist_base, signs)]

        # calibrate
        method.set_Mlist(sample_Mlist, weights=weights)
        result = method.calibrate()
        run_time += result.run_time
        if not result.success:
            continue

        # compare costs
        if best_result is None or result.aux_data['cost'] < best_result.aux_data['cost']:
            best_result = result

    # check if any calibration had success
    if best_result is None:
        return PairCalibrationResult()

    # normalize Mlist
    Mlist_norm, z = normalize_Mlist(Mlist_base, best_result.calib, frame_ids=frame_ids)

    # calibrate with updated Mlist
    method.set_Mlist(Mlist_norm, weights=weights)
    result = method.calibrate(x0=z)
    result.run_time += run_time
    return result


def calibrate_herw_sign_init(init_method: HERWCalibrationBase, main_method: HERWCalibrationBase):
    # call init method
    init_result = init_method.calibrate()
    if not init_result.success:
        return PairCalibrationResult()

    # normalize Mlist
    Mlist_norm, z = normalize_Mlist(main_method.Mlist, init_result.calib)

    # calibrate with updated Mlist
    main_method.set_Mlist(Mlist_norm)
    result = main_method.calibrate(x0=z)
    result.run_time += init_result.run_time
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
    run_time = 0.0
    for d in data:
        # optimize
        frame_ids = FrameIds(x=[d.frame_x], y=[d.frame_y])
        sub_result = calibrate_herw_sign_sampling(method, d.transforms_a, d.transforms_b, frame_ids=frame_ids,
                                                  weights=d.weights, n_reps=n_reps, n_samples=n_samples)
        run_time += sub_result.run_time
        if not sub_result.success:
            continue

        # store
        if d.frame_x not in calib_data_x or len(d.transforms_a) > calib_data_x[d.frame_x].n_samples:
            calib_data_x[d.frame_x] = _CalibData(transform=sub_result.calib.x[d.frame_x], n_samples=len(d.transforms_a))
        if d.frame_y not in calib_data_y or len(d.transforms_a) > calib_data_y[d.frame_y].n_samples:
            calib_data_y[d.frame_y] = _CalibData(transform=sub_result.calib.y[d.frame_y], n_samples=len(d.transforms_a))

    # calibration
    calib = MultiTransformPair(
        x={frame: data.transform for frame, data in calib_data_x.items()},
        y={frame: data.transform for frame, data in calib_data_y.items()}
    )

    # add all transformations to method and normalize Mlist
    method.set_transform_data(data)
    Mlist_norm, z = normalize_Mlist(method.Mlist, calib, method.frame_ids)

    # calibrate with updated Mlist
    method.set_Mlist(Mlist_norm)
    result = method.calibrate(x0=z)
    result.run_time += run_time
    return result


def normalize_Mlist(Mlist: List[np.ndarray], calib: Union[TransformPair, MultiTransformPair],
                    frame_ids: Optional[FrameIds] = None):
    # check if Mlist is list of lists and normalize separately
    assert len(Mlist) > 0
    if isinstance(Mlist[0], list):
        norm_results = [normalize_Mlist(tmp, calib, frame_ids) for tmp in Mlist]
        Mlist_norm = [x[0] for x in norm_results]
        z = norm_results[0][1]
        return Mlist_norm, z

    # generate z from a priori calibration
    if frame_ids is None:
        transforms = [calib.x, calib.y]
        num_frames_x = 1
    else:
        transforms_x = [calib.x[frame] for frame in frame_ids.x] if isinstance(calib.x, dict) else [calib.x]
        transforms_y = [calib.y[frame] for frame in frame_ids.y] if isinstance(calib.y, dict) else [calib.y]
        transforms = [*transforms_x, *transforms_y]
        num_frames_x = len(frame_ids.x)

    z = np.concatenate([t.asType(m3d.TransformType.kDualQuaternion).toArray() for t in transforms])

    # normalize Mlist based on z
    Mlist_norm = []
    for M1 in Mlist:
        M2 = M1.copy()
        M2[:, :8 * num_frames_x] *= -1.0
        if z.T @ M1.T @ M1 @ z <= z.T @ M2.T @ M2 @ z:
            Mlist_norm.append(M1)
        else:
            Mlist_norm.append(M2)

    return Mlist_norm, z
