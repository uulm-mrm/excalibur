import unittest

import excalibur as excal

import data


class TestHERWCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('Dornaika'),
                              excal.calibration.herw.Dornaika)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('DualQuaternionQCQPSignSampling'),
                              excal.calibration.herw.DualQuaternionQCQPSignSampling)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('LiDQSignSampling'),
                              excal.calibration.herw.LiDQSignSampling)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('LiHM'),
                              excal.calibration.herw.LiHM)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('Shah'),
                              excal.calibration.herw.Shah)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('Tabb'),
                              excal.calibration.herw.Tabb)
        self.assertIsInstance(excal.calibration.HERWCalibrationBase.create('Wang'),
                              excal.calibration.herw.Wang)

    def _test_method(self, name, n_random=None, **kwargs):
        # create data
        poses_a, poses_b, calib_x, calib_y = data.get_hew_pose_data(n_random=n_random)

        # estimate
        method = excal.calibration.HERWCalibrationBase.create(name)
        method.configure(**kwargs)
        method.set_transforms(poses_a, poses_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.get_messages())

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
        self._test_method('LiHM')

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQPSignSampling')

    def test_shah(self):
        self._test_method('Shah')

    def test_tabb(self):
        self._test_method('Tabb', use_cost2=False)
        self._test_method('Tabb', use_cost2=True)

    def test_wang(self):
        self._test_method('Wang')
