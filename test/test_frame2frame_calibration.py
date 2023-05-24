import unittest

import excalibur as excal

import data


class TestFrame2FrameCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(excal.calibration.Frame2FrameCalibrationBase.create('DualQuaternionQCQP'),
                              excal.calibration.frame2frame.DualQuaternionQCQP)

    def _test_method(self, name, n_random=None):
        # create data
        poses_a, poses_b, calib = data.get_target_data(n_random=n_random)

        # estimate
        method = excal.calibration.Frame2FrameCalibrationBase.create(name)
        method.set_transforms(poses_a, poses_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.get_messages())

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQP')
