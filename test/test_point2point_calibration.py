import math
import unittest

import excalibur.calibration as ec

import data


class TestPoint2PointCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(ec.Point2PointCalibrationBase.create('Arun'),
                              ec.point2point.Arun)
        self.assertIsInstance(ec.Point2PointCalibrationBase.create('DualQuaternionQCQP'),
                              ec.point2point.DualQuaternionQCQP)
        self.assertIsInstance(ec.Point2PointCalibrationBase.create('HornQuat'),
                              ec.point2point.HornQuat)
        self.assertIsInstance(ec.Point2PointCalibrationBase.create('MatrixQCQP'),
                              ec.point2point.MatrixQCQP)

    def _test_method(self, name, init_kwargs=None, **kwargs):
        if init_kwargs is None:
            init_kwargs = {}

        # create data
        poses_a, poses_b, calib = data.get_target_data()
        points_a = data.get_target_points(poses_a)
        points_b = data.get_target_points(poses_b)

        # estimate
        method = ec.Point2PointCalibrationBase.create(name, **init_kwargs)
        method.configure(**kwargs)
        method.set_points(points_a, points_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.message)

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

    def test_ransac(self):
        self._test_method('Point2PointRANSAC', init_kwargs={
            'method_name': 'DualQuaternionQCQP',
            'nreps': 10,
            'trans_thresh': math.inf,
        })
