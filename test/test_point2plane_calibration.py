import unittest

import excalibur.calibration as ec

from . import data


class TestPoint2PlaneCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(ec.Point2PlaneCalibrationBase.create('MatrixQCQP'),
                              ec.point2plane.MatrixQCQP)

    def _test_method(self, name, init_kwargs=None, **kwargs):
        if init_kwargs is None:
            init_kwargs = {}

        # create data
        points_a, planes_b, calib = data.get_point2plane_data()

        # estimate
        method = ec.Point2PlaneCalibrationBase.create(name, **init_kwargs)
        method.configure(**kwargs)
        method.set_data(points_a, planes_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.message)

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_qcqp_hm(self):
        self._test_method('MatrixQCQP')
