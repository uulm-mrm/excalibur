from dataclasses import dataclass, field
import time
from typing import Dict, Optional
import warnings

import motion3d as m3d
import numpy as np

import excalibur as excal
from excalibur.calibration.herw.dq.sign_sampling import calibrate_herw_sign_sampling, calibrate_herw_sign_sampling_multi
from excalibur.metrics.infrastructure import line_projection_errors
from excalibur.utils.image import project_cc_to_ic, solve_pnp


def _generate_chessboard_world_points(chess_dim, square_size):
    object_pts = np.zeros((chess_dim[0] * chess_dim[1], 3), np.float32)
    object_pts[:, :2] = np.mgrid[0:chess_dim[0], 0:chess_dim[1]].T.reshape(-1, 2)
    object_pts *= square_size

    object_pts -= np.array([(chess_dim[0] - 1) * square_size / 2,
                            (chess_dim[1] - 1) * square_size / 2, 0])
    object_pts[:, 1] *= -1
    return object_pts


CHECKERBOARD = _generate_chessboard_world_points((5, 3), 0.24)


@dataclass
class MethodConfig:
    name: str
    init_kwargs: Optional[Dict] = field(default_factory=dict)
    calib_kwargs: Optional[Dict] = field(default_factory=dict)
    force_positive_z: bool = False


def _handle_error_result(method_name, result):
    print(f"[{method_name}] {result.get_messages()}")
    return {'t_err_x': None,
            'r_err_x': None,
            't_err_y': None,
            'r_err_y': None,
            'time': None}


def _transformation_errors(pred, calib):
    if pred is None or calib is None:
        return None, None, None

    calib_err = pred / calib
    t_err = calib_err.translationNorm()
    r_err = calib_err.rotationNorm()
    t_diff = np.linalg.norm(pred.asType(m3d.TransformType.kQuaternion).getTranslation() -
                            calib.asType(m3d.TransformType.kQuaternion).getTranslation())
    return t_err, r_err, t_diff


def _transformation_errors_f2f(pred_y, calib_y):
    t_err_y, r_err_y, t_diff_y = _transformation_errors(pred_y, calib_y)
    return {'t_err_x': None,
            't_err_y': t_err_y,
            'r_err_x': None,
            'r_err_y': r_err_y,
            't_diff_x': None,
            't_diff_y': t_diff_y}


def _transformation_errors_herw(pred_x, pred_y, calib_x, calib_y):
    t_err_x, r_err_x, t_diff_x = _transformation_errors(pred_x, calib_x)
    t_err_y, r_err_y, t_diff_y = _transformation_errors(pred_y, calib_y)
    return {'t_err_x': t_err_x,
            't_err_y': t_err_y,
            'r_err_x': r_err_x,
            'r_err_y': r_err_y,
            't_diff_x': t_diff_x,
            't_diff_y': t_diff_y}


def _cycle_errors_herw(transforms_a, transforms_b, calib_x, calib_y):
    r_errs = []
    t_errs = []
    dq_errs = []
    for (pa, pb) in zip(transforms_a, transforms_b):
        cycle = (calib_y * pb).inverse() * (pa * calib_x)
        r_errs.append(cycle.rotationNorm())
        t_errs.append(cycle.translationNorm())
        dq_err1 = np.linalg.norm(cycle.asType(m3d.TransformType.kDualQuaternion).toArray() -
                                 m3d.DualQuaternionTransform().toArray())
        dq_err2 = np.linalg.norm(cycle.asType(m3d.TransformType.kDualQuaternion).toArray() +
                                 m3d.DualQuaternionTransform().toArray())
        dq_errs.append(min(dq_err1, dq_err2))
    return {'r_errs_cycle': r_errs, 't_errs_cycle': t_errs, 'dq_errors': dq_errs}


def _reprojection_errors_squared(transforms_a, transforms_b, calib_x, calib_y,
                                 target_points, intrinsics, image_pts=None):
    # check if target points are 3D
    if target_points.shape[1] == 2:
        target_points_3d = np.column_stack((target_points, np.zeros(target_points.shape[0])))
    elif target_points.shape[1] == 3:
        target_points_3d = target_points
    else:
        raise RuntimeError("Invalid target points")

    reprojection_errors_squared = []
    for i, (ta, tb) in enumerate(zip(transforms_a, transforms_b)):
        # measured points in image coordiates
        if image_pts is None:
            target2cam_meas = tb
            target_meas_cc = target2cam_meas.transformCloud(target_points_3d.T).T
            target_meas_ic = project_cc_to_ic(target_meas_cc, intrinsics)
        else:
            target_meas_ic = image_pts[:, :, i]

        # projected target points in image coordinates
        target2cam_est = calib_y.inverse() * ta * calib_x
        target_est_cc = target2cam_est.transformCloud(target_points_3d.T).T
        target_est_ic = project_cc_to_ic(target_est_cc, intrinsics)

        # append error
        reprojection_errors_squared.append(np.sum(np.square(target_meas_ic - target_est_ic), axis=1))
    return reprojection_errors_squared


def _reprojection_error(transforms_a, transforms_b, calib_x, calib_y, target_points, intrinsics, image_pts=None):
    errors_squared = _reprojection_errors_squared(transforms_a, transforms_b, calib_x, calib_y, target_points,
                                                  intrinsics, image_pts=image_pts)
    return np.sqrt(np.mean(errors_squared))


def _reprojection_errors_multi(transform_data, calibs_x, calibs_y, target_points, intrinsics, frames_x=None):
    # initialize output
    reprojection_errors_squared = {}

    # iterate samples
    for d in transform_data:
        # check x frame
        if frames_x is not None and d.frame_x not in frames_x:
            continue

        # get calib
        calib_x = calibs_x[d.frame_x]
        calib_y = calibs_y[d.frame_y]
        intrinsics_y = intrinsics[d.frame_y]

        # calculate and store errors
        errors_squared = _reprojection_errors_squared(d.transforms_a, d.transforms_b, calib_x, calib_y, target_points,
                                                      intrinsics_y)
        if d.frame_y not in reprojection_errors_squared:
            reprojection_errors_squared[d.frame_y] = errors_squared
        else:
            reprojection_errors_squared[d.frame_y].extend(errors_squared)

    # average
    reprojection_errors = {frame: np.sqrt(np.mean(errors_squared))
                           for frame, errors_squared in reprojection_errors_squared.items()}
    return reprojection_errors


def _check_herw_dq_sample_costs(method, result):
    x = result.aux_data['x']
    for M in method.Mlist:
        trafo_indices = []
        for trafo_index in range(int(M.shape[1]/8)):
            if np.sum(np.abs(M[:, trafo_index*8:(trafo_index+1)*8])) > 0:
                trafo_indices.append(trafo_index)
        assert len(trafo_indices) == 2
        a_index, b_index = trafo_indices[0], trafo_indices[1]

        M_alt = M.copy()
        M_alt[:, b_index*8:(b_index+1)*8] *= -1

        costs = x.T @ M.T @ M @ x
        costs_alt = x.T @ M_alt.T @ M_alt @ x
        if costs > costs_alt:
            return False
    return True


def _calibrate_pnp(transforms_a, transforms_b, ground_truth_x, intrinsics):
    # generate points for PnP
    pts_cam = []
    pts_world = []

    for ta, tb in zip(transforms_a, transforms_b):
        # target in image coordinates
        target_meas_cc = tb.transformCloud(CHECKERBOARD.T).T
        target_meas_ic = project_cc_to_ic(target_meas_cc, intrinsics)
        pts_cam.append(target_meas_ic.squeeze())

        # target in world coordinates
        target_wc = (ta * ground_truth_x).transformCloud(CHECKERBOARD.T).T
        pts_world.append(target_wc)

    pts_cam = np.row_stack(pts_cam)
    pts_world = np.row_stack(pts_world)

    # run PnP
    t_start = time.time()
    transform = solve_pnp(pts_world, pts_cam, intrinsics).inverse_()
    run_time = time.time() - t_start

    # create result
    result = excal.calibration.PairCalibrationResult()
    result.success = True
    result.calib = excal.calibration.TransformPair(
        x=ground_truth_x,
        y=transform
    )
    result.run_time = run_time
    return result


def _calibrate_f2f(method_config, transforms_a, transforms_b, ground_truth_x):
    # adjust transforms
    transforms_a_mod = transforms_a.applyPost(ground_truth_x)

    # initialize method and set transforms
    method = excal.calibration.frame2frame.DualQuaternionQCQP(**method_config.init_kwargs)
    method.configure(**method_config.calib_kwargs)
    method.set_transforms(transforms_a_mod, transforms_b)

    # calibrate
    f2f_result = method.calibrate()

    # create herw result
    result = excal.calibration.PairCalibrationResult()
    result.success = f2f_result.success
    result.calib = excal.calibration.TransformPair(
        x=ground_truth_x,
        y=f2f_result.calib
    )
    result.run_time = f2f_result.run_time
    result.aux_data = f2f_result.aux_data
    return result


def _rmse(x):
    return np.sqrt(np.mean(np.square(x)))


def calibrate_herw(method_config: MethodConfig, transforms_a, transforms_b, ground_truth_x, ground_truth_y,
                   intrinsics=None, lines=None, detections=None, return_calib=False):
    if method_config.name == 'F2F':
        result = _calibrate_f2f(method_config, transforms_a, transforms_b, ground_truth_x)
    elif method_config.name == 'PnP':
        result = _calibrate_pnp(transforms_a, transforms_b, ground_truth_x, intrinsics)
    else:
        # initialize method and set transforms
        method = excal.calibration.HERWCalibrationBase.create(method_config.name, **method_config.init_kwargs)
        method.configure(**method_config.calib_kwargs)
        method.set_transforms(transforms_a, transforms_b)

        # calibrate
        result = method.calibrate()

    # check result
    if not result.success:
        return _handle_error_result(method_config.name, result)

    # check DQ sample costs
    if method_config.name.startswith('DualQuaternionQCQP'):
        if not _check_herw_dq_sample_costs(method, result):
            warnings.warn("RANSAC for DQ sign probably did not work")

    # force positive z
    if method_config.force_positive_z:
        result = excal.calibration.herw.force_positive_z(result, transforms_a)

    # return calibration if required
    if return_calib:
        return result

    # base metrics
    metrics = {}
    metrics.update(_transformation_errors_herw(result.calib.x, result.calib.y, ground_truth_x, ground_truth_y))
    metrics.update(_cycle_errors_herw(transforms_a, transforms_b, result.calib.x, result.calib.y))
    metrics['time'] = result.run_time
    metrics['gap'] = result.aux_data['gap'] if 'gap' in result.aux_data else None

    # line error
    metrics['rel_line_errors'] = None
    if lines is not None and intrinsics is not None:
        # estimate ground plane
        ground_points = np.array([t.getTranslation() for t in transforms_a])
        ground_plane_wc = excal.fitting.plane.fit_plane(ground_points)

        # calculate line projection errors
        metrics['rel_line_error'] = _rmse(line_projection_errors(result.calib.y, ground_plane_wc, intrinsics, lines)[1])

    # reprojection error
    metrics['reprojection_error'] = None
    if detections is not None:
        img_pts = detections['img_pts']
        world_pts = detections['world_pts']
        metrics['reprojection_error'] = _reprojection_error(transforms_a, transforms_b, result.calib.x, result.calib.y,
                                                            world_pts, intrinsics, img_pts)
    elif intrinsics is not None:
        metrics['reprojection_error'] = _reprojection_error(transforms_a, transforms_b, result.calib.x, result.calib.y,
                                                            CHECKERBOARD, intrinsics)

    return metrics


def calibrate_herw_multi(method_config: MethodConfig, transform_data, ground_truths_x, ground_truths_y,
                         lines=None, intrinsics=None, return_calib=False):
    # initialize method and set transforms
    method = excal.calibration.HERWCalibrationBase.create(method_config.name, **method_config.init_kwargs)
    method.configure(**method_config.calib_kwargs)
    method.set_transform_data(transform_data)

    # calibrate
    result = method.calibrate()

    # check result
    if not result.success:
        return _handle_error_result(method_config.name, result)

    # check DQ sample costs
    if method_config.name.startswith('DualQuaternionQCQP'):
        if not _check_herw_dq_sample_costs(method, result):
            warnings.warn("RANSAC for DQ sign probably did not work")

    # force positive z
    if method_config.force_positive_z:
        result = excal.calibration.herw.force_positive_z_multi(result, transform_data)

    # return calibration if required
    if return_calib:
        return result

    # base metrics
    metrics = {'t_errs_x': {}, 'r_errs_x': {}, 't_errs_y': {}, 'r_errs_y': {}}

    for frame, calib in result.calib.x.items():
        if frame not in ground_truths_x:
            continue
        t_err, r_err, _ = _transformation_errors(calib, ground_truths_x[frame])
        metrics['t_errs_x'][frame] = t_err
        metrics['r_errs_x'][frame] = r_err
    for frame, calib in result.calib.y.items():
        if frame not in ground_truths_y:
            continue
        t_err, r_err, _ = _transformation_errors(calib, ground_truths_y[frame])
        metrics['t_errs_y'][frame] = t_err
        metrics['r_errs_y'][frame] = r_err

    metrics['time'] = result.run_time
    metrics['gap'] = result.aux_data['gap'] if 'gap' in result.aux_data else None

    # line errors
    metrics['rel_line_errors'] = {}
    if lines is not None and intrinsics is not None:
        for frame_y, calib in result.calib.y.items():
            # estimate ground plane
            ground_points = np.row_stack([[t.getTranslation() for t in d.transforms_a]
                                          for d in transform_data if d.frame_y == frame_y])
            ground_plane_wc = excal.fitting.plane.fit_plane(ground_points)

            # calculate line projection errors
            metrics['rel_line_errors'][frame_y] = \
                _rmse(line_projection_errors(calib, ground_plane_wc, intrinsics[frame_y], lines[frame_y])[1])

    # reprojection errors
    metrics['reprojection_errors'] = {}
    if intrinsics is not None:
        metrics['reprojection_errors'] = _reprojection_errors_multi(
            transform_data, result.calib.x, result.calib.y, CHECKERBOARD, intrinsics, ['chessboard'])

    return metrics
