from dataclasses import dataclass
import time
from typing import Optional

import numpy as np
import scipy.optimize

from excalibur.utils.logging import logger


@dataclass
class LinResult:
    success: bool = False
    message: str = ""
    run_time: Optional[float] = None
    opt_result: Optional[scipy.optimize.OptimizeResult] = None
    value: Optional[float] = None
    x: Optional[np.ndarray] = None


def solve_linear_problem(A: np.ndarray, b: np.ndarray) -> LinResult:
    # initialize result
    result = LinResult()

    # optimize
    start_time = time.time()
    result.opt_result = scipy.optimize.lsq_linear(A, b.squeeze())
    result.run_time = time.time() - start_time

    # store results
    result.success = result.opt_result.success
    if result.opt_result.success:
        result.value = result.opt_result.cost
        result.x = result.opt_result.x
    else:
        result.message = result.opt_result.message
        logger.warning(result.message)

    return result
