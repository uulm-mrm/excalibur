import unittest

import excalibur as excal

import data


class TestPoint2PointCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(excal.calibration.Point2PointCalibrationBase.create('Arun'),
                              excal.calibration.point2point.Arun)
        self.assertIsInstance(excal.calibration.Point2PointCalibrationBase.create('DualQuaternionQCQP'),
                              excal.calibration.point2point.DualQuaternionQCQP)
        self.assertIsInstance(excal.calibration.Point2PointCalibrationBase.create('HornQuat'),
                              excal.calibration.point2point.HornQuat)
        self.assertIsInstance(excal.calibration.Point2PointCalibrationBase.create('MatrixQCQP'),
                              excal.calibration.point2point.MatrixQCQP)

    def _test_method(self, name):
        # create data
        poses_a, poses_b, calib = data.get_target_data()
        points_a = data.get_target_points(poses_a)
        points_b = data.get_target_points(poses_b)

        # estimate
        method = excal.calibration.Point2PointCalibrationBase.create(name)
        method.set_points(points_a, points_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.get_messages())

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_arun(self):
        self._test_method('Arun')

    def test_horn_quat(self):
        self._test_method('HornQuat')

    def test_qcqp_hm(self):
        self._test_method('MatrixQCQP')

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQP')
