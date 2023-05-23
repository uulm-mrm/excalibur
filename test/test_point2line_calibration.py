import unittest

import numpy as np

import excalibur as excal

import data


class TestPoint2LineCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(excal.calibration.Point2LineCalibrationBase.create('MatrixQCQP'),
                              excal.calibration.point2line.MatrixQCQP)

    def _test_method(self, name, with_line_origins):
        # create data
        poses_a, poses_b, calib = data.get_target_data()
        points_a = data.get_target_points(poses_a)
        points_b = data.get_target_points(poses_b)

        if with_line_origins:
            line_origins_b = -points_b + 1.0
            line_vecs_b = points_b - line_origins_b
            line_vecs_b /= np.linalg.norm(line_vecs_b, axis=0)
        else:
            line_origins_b = None
            line_vecs_b = points_b / np.linalg.norm(points_b, axis=0)

        # estimate
        method = excal.calibration.Point2LineCalibrationBase.create(name)
        method.set_data(points_a, line_vecs_b, line_origins_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.get_messages())

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_qcqp_hm(self):
        self._test_method('MatrixQCQP', with_line_origins=False)
        self._test_method('MatrixQCQP', with_line_origins=True)
