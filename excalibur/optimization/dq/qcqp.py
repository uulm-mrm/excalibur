from typing import List, Union

import numpy as np

from ..qcqp import DualRecoveryResult, QCQPDualResult, QCQPRelaxedResult, QuadraticFun, SDRRecoveryResult, _QCQPProblem
from ..recovery import calculate_nullspace_factors_1d, recover_from_dual, recover_from_sdr

from .recovery import calculate_nullspace_factors_daniilidis


class QCQPProblemDQ(_QCQPProblem):
    def __init__(self, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                 real_indices: Union[List, np.ndarray], dual_indices: Union[List, np.ndarray]):
        super().__init__(Q, constraint_funs)
        self.real_indices = real_indices
        self.dual_indices = dual_indices

    def recover_from_dual(self, dual_result: QCQPDualResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                          **kwargs) -> DualRecoveryResult:
        nullspace_factor_fun_dict = {
            1: lambda ns: calculate_nullspace_factors_1d(ns, self.real_indices),
            2: lambda ns: calculate_nullspace_factors_daniilidis(ns, self.real_indices, self.dual_indices),
        }
        return recover_from_dual(dual_result, Q, constraint_funs, nullspace_factor_fun_dict, **kwargs)

    def recover_from_sdr(self, relaxed_result: QCQPRelaxedResult, Q: np.ndarray, constraint_funs: List[QuadraticFun],
                         **kwargs) -> SDRRecoveryResult:
        return recover_from_sdr(relaxed_result, Q, constraint_funs, norm_one_indices=self.real_indices, **kwargs)
