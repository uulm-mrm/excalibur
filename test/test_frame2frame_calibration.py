import math
import unittest

import excalibur.calibration as ec

from . import data


class TestFrame2FrameCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(ec.Frame2FrameCalibrationBase.create('DualQuaternionQCQP'),
                              ec.frame2frame.DualQuaternionQCQP)

    def _test_method(self, name, n_random=None, init_kwargs=None, **kwargs):
        if init_kwargs is None:
            init_kwargs = {}

        # create data
        poses_a, poses_b, calib = data.get_target_data(n_random=n_random)

        # estimate
        method = ec.Frame2FrameCalibrationBase.create(name, **init_kwargs)
        method.configure(**kwargs)
        method.set_transforms(poses_a, poses_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.message)

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQP')

    def test_ransac(self):
        self._test_method('Frame2FrameRANSAC', init_kwargs={
            'method_name': 'DualQuaternionQCQP',
            'nreps': 10,
            'rot_thresh': math.inf,
            'trans_thresh': math.inf,
        })
