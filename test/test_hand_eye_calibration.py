import unittest

import excalibur as excal

import data


class TestHandEyeCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('Andreff'),
                              excal.calibration.hand_eye.Andreff)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('Daniilidis'),
                              excal.calibration.hand_eye.Daniilidis)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('DualQuaternionQCQP'),
                              excal.calibration.hand_eye.DualQuaternionQCQP)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('DualQuaternionQCQPScaled'),
                              excal.calibration.hand_eye.DualQuaternionQCQPScaled)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('MatrixQCQP'),
                              excal.calibration.hand_eye.MatrixQCQP)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('MatrixQCQPScaled'),
                              excal.calibration.hand_eye.MatrixQCQPScaled)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('SchmidtDQ'),
                              excal.calibration.hand_eye.SchmidtDQ)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('SchmidtHM'),
                              excal.calibration.hand_eye.SchmidtHM)
        self.assertIsInstance(excal.calibration.HandEyeCalibrationBase.create('Wei'),
                              excal.calibration.hand_eye.Wei)

    def _test_method(self, name, **kwargs):
        # create data
        motion_a, motion_b, calib = data.get_motion_data()

        # estimate
        method = excal.calibration.HandEyeCalibrationBase.create(name)
        method.configure(**kwargs)
        method.set_transforms(motion_a, motion_b)
        result = method.calibrate()

        # check success
        if not result.success:
            self.fail(result.get_messages())

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_andreff(self):
        self._test_method('Andreff')

    def test_daniilidis(self):
        self._test_method('Daniilidis')

    def test_qcqp_hm(self):
        self._test_method('MatrixQCQP')
        self._test_method('MatrixQCQPScaled')

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQP')
        self._test_method('DualQuaternionQCQPScaled')

    def test_schmidt(self):
        self._test_method('SchmidtDQ', improved=False)
        self._test_method('SchmidtDQ', improved=True)
        self._test_method('SchmidtHM', improved=False)
        self._test_method('SchmidtHM', improved=True)

    def test_wei(self):
        self._test_method('Wei')
