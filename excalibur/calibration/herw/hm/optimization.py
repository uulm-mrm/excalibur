import time

import motion3d as m3d
import numpy as np
import scipy.optimize
import transforms3d as t3d

from excalibur.utils.logging import MessageLevel, Message
from excalibur.utils.parameters import add_default_kwargs
from ...base import MultiTransformPair, PairCalibrationResult, TransformPair


def _init_dornaika(calib_init=None, normalize=True):
    # prepare
    if calib_init is None:
        transform_x = m3d.MatrixTransform()
        transform_y = m3d.MatrixTransform()
    else:
        transform_x = calib_init.x.asType(m3d.TransformType.kMatrix)
        transform_y = calib_init.y.asType(m3d.TransformType.kMatrix)
        if normalize:
            transform_x.normalized_()
            transform_y.normalized_()

    # convert
    return np.concatenate([transform_x.getRotationMatrix().reshape(9), transform_x.getTranslation(),
                           transform_y.getRotationMatrix().reshape(9), transform_y.getTranslation()])


def _split_dornaika(x):
    Rx = x[:9].reshape(3, 3)
    tx = x[9:12].reshape(3, 1)
    Ry = x[12:21].reshape(3, 3)
    ty = x[21:24].reshape(3, 1)
    return Rx, tx, Ry, ty


def _cost_dornaika(x, data, weights):
    # split x
    Rx, tx, Ry, ty = _split_dornaika(x)

    # split data
    Ra = data.A[:, :3, :3]
    Rb = data.B[:, :3, :3]
    ta = data.A[:, :3, 3]
    tb = data.B[:, :3, 3]

    # cycle errors
    rot_errors = Ra @ Rx - (Rb.transpose(0, 2, 1) @ Ry.T).transpose(0, 2, 1)
    trans_errors = Ra @ tx.squeeze() + ta - (Ry @ tb.T).T - ty.T

    # combine with constraint costs
    errors = np.concatenate((
        weights[0] * rot_errors.flatten(),
        weights[1] * trans_errors.flatten(),
        weights[2] * (Rx @ Rx.T - np.eye(3)).reshape(9),
        weights[3] * (Ry @ Ry.T - np.eye(3)).reshape(9)
    ))
    return errors


def optimize_dornaika(matrix_data, calib_init=None, weights=None, solver_kwargs=None):
    # initialize result
    result = PairCalibrationResult()

    # initial solution
    x0 = _init_dornaika(calib_init)

    # weights
    if weights is None:
        weights = np.array([1.0, 1.0, 1e6, 1e6])
    weights = np.sqrt(weights)

    # solver arguments
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        method='lm',
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=200 * len(x0),
    )

    # optimization
    start_time = time.time()
    lsq_result = scipy.optimize.least_squares(
        lambda x: _cost_dornaika(x, matrix_data, weights),
        x0,
        **solver_kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not lsq_result.success:
        result.msgs.append(Message(text=f"Optimization failed: {lsq_result.message}",
                                   level=MessageLevel.FATAL))
        return result

    # construct transforms
    x_est = lsq_result.x
    Rx, tx, Ry, ty = _split_dornaika(x_est)
    solution_x = m3d.MatrixTransform(tx, Rx, unsafe=True).normalized_()
    solution_y = m3d.MatrixTransform(ty, Ry, unsafe=True).normalized_()

    # create result
    result.success = True
    result.calib = TransformPair(x=solution_x, y=solution_y)
    result.aux_data = {
        'lsq_result': lsq_result
    }
    return result


def _gen_tabb_init(calib_init, frame_ids, normalize=True):
    # prepare
    if calib_init is None:
        if frame_ids is None:
            transforms_x = [m3d.QuaternionTransform()]
            transforms_y = [m3d.QuaternionTransform()]
        else:
            transforms_x = [m3d.QuaternionTransform() for _ in frame_ids.x]
            transforms_y = [m3d.QuaternionTransform() for _ in frame_ids.y]
    else:
        if isinstance(calib_init, TransformPair):
            transforms_x = [calib_init.x.asType(m3d.TransformType.kQuaternion)]
            transforms_y = [calib_init.y.asType(m3d.TransformType.kQuaternion)]
        else:
            assert frame_ids is not None
            transforms_x = [calib_init.x[frame].asType(m3d.TransformType.kQuaternion) for frame in frame_ids.x]
            transforms_y = [calib_init.y[frame].asType(m3d.TransformType.kQuaternion) for frame in frame_ids.y]

        # normalize
        if normalize:
            transforms_x = [t.normalized_() for t in transforms_x]
            transforms_y = [t.normalized_() for t in transforms_y]

    # convert
    return np.concatenate([*[t.toArray() for t in transforms_x],
                           *[t.toArray() for t in transforms_y]])


def _cost_tabb(vec, matrix_data_list, use_cost2=False):
    # convert x
    vec_x = vec[:7]
    if use_cost2:
        X = m3d.QuaternionTransform(vec_x, unsafe=True).asType(m3d.TransformType.kMatrix).inverse().getMatrix()
    else:
        X = m3d.QuaternionTransform(vec_x, unsafe=True).asType(m3d.TransformType.kMatrix).getMatrix()

    # iterate y
    costs = []
    for data in matrix_data_list:
        # convert Y
        vec_y = vec[7 * (data.y_idx + 1):7 * (data.y_idx + 2)]
        Y = m3d.QuaternionTransform(vec_y, unsafe=True).asType(m3d.TransformType.kMatrix).getMatrix()

        # residuals
        if use_cost2:
            residuals = data.A - (data.B.transpose(0, 2, 1) @ Y.T).transpose(0, 2, 1) @ X
        else:
            residuals = data.A @ X - (data.B.transpose(0, 2, 1) @ Y.T).transpose(0, 2, 1)
        costs.extend(residuals.flatten())
    return costs


def optimize_tabb(matrix_data_list, frame_ids, calib_init=None, use_cost2=False, solver_kwargs=None):
    assert len(matrix_data_list) > 0

    # initialize result
    result = PairCalibrationResult()

    # get x and y count
    x_count = len(frame_ids.x) if frame_ids is not None else 1
    y_count = len(frame_ids.y) if frame_ids is not None else 1
    assert x_count == 1

    # initial solution
    x0 = _gen_tabb_init(calib_init, frame_ids)

    # solver arguments
    solver_kwargs = add_default_kwargs(
        solver_kwargs,
        method='lm',
    )

    # optimization
    start_time = time.time()
    lsq_result = scipy.optimize.least_squares(
        lambda x: _cost_tabb(x, matrix_data_list, use_cost2=use_cost2),
        x0,
        **solver_kwargs)
    result.run_time = time.time() - start_time

    # check success
    if not lsq_result.success:
        result.aux_data = {'lsq_result': lsq_result}
        result.msgs.append(Message(text=f"Optimization failed: {lsq_result.message}",
                                   level=MessageLevel.FATAL))
        return result

    # construct transforms
    x_est = lsq_result.x
    if frame_ids is None:
        solution_x = m3d.QuaternionTransform(x_est[:7], unsafe=True).normalized_()
        solution_y = m3d.QuaternionTransform(x_est[7:], unsafe=True).normalized_()
        result.calib = MultiTransformPair(x=solution_x, y=solution_y)
    else:
        solutions_x = {
            frame_ids.x[x_id]:
                m3d.QuaternionTransform(x_est[7 * x_id:7 * (x_id + 1)],
                                        unsafe=True).normalized_()
            for x_id in range(x_count)
        }
        solutions_y = {
            frame_ids.y[y_id]:
                m3d.QuaternionTransform(x_est[7 * (len(frame_ids.x) + y_id):7 * (len(frame_ids.x) + y_id + 1)],
                                        unsafe=True).normalized_()
            for y_id in range(y_count)
        }
        result.calib = MultiTransformPair(x=solutions_x, y=solutions_y)

    # result
    result.success = True
    result.aux_data = {
        'opt_result': lsq_result
    }
    return result
