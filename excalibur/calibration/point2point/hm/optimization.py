import time

import motion3d as m3d
import numpy as np

from ...base import CalibrationResult
from excalibur.optimization.hm import recover_from_dual, rotmat_constraints_hom
from excalibur.optimization.qcqp import solve_qcqp_dual
from excalibur.utils.logging import MessageLevel, Message
from excalibur.utils.math import schur_complement_indices, submat
from excalibur.utils.parameters import add_default_kwargs


# indices and constraints
DIM = 13
T_INDICES = [9, 10, 11]
NT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]
HOM_INDEX_RED = -1

# constraints
RM_CONSTRAINTS = rotmat_constraints_hom()


def optimize(Q, solver_kwargs=None, recovery_kwargs=None):
    # check input
    assert Q.shape == (DIM, DIM)

    # default arguments
    solver_kwargs = add_default_kwargs(solver_kwargs)
    recovery_kwargs = add_default_kwargs(recovery_kwargs)

    # initialize result
    result = CalibrationResult()

    # start time measurement
    start_time = time.time()

    # reduce Q
    Q_red = schur_complement_indices(Q, NT_INDICES, NT_INDICES, T_INDICES, T_INDICES)

    # solve dual problem
    dual_result = solve_qcqp_dual(Q_red, RM_CONSTRAINTS, **solver_kwargs)

    # check success
    if not dual_result.success:
        result.msgs.append(Message(text=f"Optimization failed",
                                   level=MessageLevel.FATAL))
        return result

    # recover primal solution
    dual_recovery = recover_from_dual(dual_result, Q_red, RM_CONSTRAINTS, HOM_INDEX_RED, **recovery_kwargs)

    if not dual_recovery.success:
        result.msgs.append(Message(text=f"Recovery failed",
                                   level=MessageLevel.FATAL))
        return result

    # construct transformation
    x_est = dual_recovery.x
    x_rotmat = x_est[:9].reshape(3, 3).T
    x_trans = - np.linalg.inv(submat(Q, T_INDICES, T_INDICES)) @ submat(Q, T_INDICES, NT_INDICES) @ x_est

    # stop time
    result.run_time = time.time() - start_time

    matrix_solution = m3d.MatrixTransform(x_trans, x_rotmat, unsafe=True)
    matrix_solution.normalized_()
    matrix_solution.inverse_()  # Briales' definition is inverse to our definition

    # result
    result.success = True
    result.calib = matrix_solution
    result.aux_data = {
        'dual_result': dual_result,
        'dual_recovery': dual_recovery,
    }
    return result
