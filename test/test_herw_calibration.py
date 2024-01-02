import math
import unittest

import excalibur.calibration as ec

from . import data


RANSAC_KWARGS = {
    'nreps': 10,
    'rot_thresh': math.inf,
    'trans_thresh': math.inf
}


class TestHERWCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(ec.HERWCalibrationBase.create('Dornaika'),
                              ec.herw.Dornaika)
        self.assertIsInstance(ec.HERWCalibrationBase.create('DualQuaternionQCQPSignSampling'),
                              ec.herw.DualQuaternionQCQPSignSampling)
        self.assertIsInstance(ec.HERWCalibrationBase.create('DualQuaternionQCQPSeparableInit'),
                              ec.herw.DualQuaternionQCQPSeparableInit)
        self.assertIsInstance(ec.HERWCalibrationBase.create('DualQuaternionQCQPSeparableRANSACInit', **RANSAC_KWARGS),
                              ec.herw.DualQuaternionQCQPSeparableRANSACInit)
        self.assertIsInstance(ec.HERWCalibrationBase.create('LiDQSignSampling'),
                              ec.herw.LiDQSignSampling)
        self.assertIsInstance(ec.HERWCalibrationBase.create('LiDQSignBruteForce'),
                              ec.herw.LiDQSignBruteForce)
        self.assertIsInstance(ec.HERWCalibrationBase.create('LiDQSignInitHM'),
                              ec.herw.LiDQSignInitHM)
        self.assertIsInstance(ec.HERWCalibrationBase.create('LiHM'),
                              ec.herw.LiHM)
        self.assertIsInstance(ec.HERWCalibrationBase.create('Shah'),
                              ec.herw.Shah)
        self.assertIsInstance(ec.HERWCalibrationBase.create('Tabb'),
                              ec.herw.Tabb)
        self.assertIsInstance(ec.HERWCalibrationBase.create('Wang'),
                              ec.herw.Wang)

    def _test_method(self, name, init_kwargs=None, **kwargs):
        if init_kwargs is None:
            init_kwargs = {}

        # create data
        poses_a, poses_b, calib_x, calib_y = data.get_herw_pose_data()

        # estimate
        method = ec.HERWCalibrationBase.create(name, **init_kwargs)
        method.configure(**kwargs)
        method.set_transforms(poses_a, poses_b)
        result = method.calibrate()

        # check
        self._check_result(result, calib_x, calib_y)

    def _check_result(self, result, calib_x, calib_y):
        # check success
        if not result.success:
            self.fail(result.message)

        # check estimate
        error_x = calib_x / result.calib.x
        self.assertAlmostEqual(error_x.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error_x.translationNorm(), 0.0, places=3)
        error_y = calib_y / result.calib.y
        self.assertAlmostEqual(error_y.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error_y.translationNorm(), 0.0, places=3)

    def test_dornaika(self):
        self._test_method('Dornaika')

    def test_li(self):
        self._test_method('LiDQSignSampling')
        self._test_method('LiDQSignBruteForce')
        self._test_method('LiDQSignInitHM')
        self._test_method('LiHM')

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQPSignSampling')
        self._test_method('DualQuaternionQCQPSeparableInit')
        self._test_method('DualQuaternionQCQPSeparableRANSACInit', init_kwargs=RANSAC_KWARGS)

    def test_qcqp_dq_norm(self):
        # create data
        poses_a, poses_b, calib_x, calib_y = data.get_herw_pose_data(planar=True)

        # translation norm constraint
        config_kwargs = {
            't_norms': [calib_x.translationNorm(), None],
            'up_approx': [0.0, 0.0, 1.0],
        }

        # estimate
        method = ec.HERWCalibrationBase.create('DualQuaternionQCQPSeparableInit')
        method.configure(**config_kwargs)
        method.set_transforms(poses_a, poses_b)
        result = method.calibrate()

        # check
        self._check_result(result, calib_x, calib_y)

    def test_shah(self):
        self._test_method('Shah')

    def test_tabb(self):
        self._test_method('Tabb', use_cost2=False)
        self._test_method('Tabb', use_cost2=True)

    def test_wang(self):
        self._test_method('Wang')

    def test_ransac(self):
        self._test_method('HERWRANSAC', init_kwargs={
            'method_name': 'DualQuaternionQCQPSeparableInit',
            **RANSAC_KWARGS
        })
