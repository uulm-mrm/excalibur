import itertools
import math
import unittest

import numpy as np

import excalibur.calibration as ec
from excalibur.fitting.plane import Plane

import data


class TestHandEyeCalibration(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('Andreff'),
                              ec.hand_eye.Andreff)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('Daniilidis'),
                              ec.hand_eye.Daniilidis)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('DualQuaternionQCQP'),
                              ec.hand_eye.DualQuaternionQCQP)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('DualQuaternionQCQPScaled'),
                              ec.hand_eye.DualQuaternionQCQPScaled)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('MatrixQCQP'),
                              ec.hand_eye.MatrixQCQP)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('MatrixQCQPScaled'),
                              ec.hand_eye.MatrixQCQPScaled)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('SchmidtDQ'),
                              ec.hand_eye.SchmidtDQ)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('SchmidtHM'),
                              ec.hand_eye.SchmidtHM)
        self.assertIsInstance(ec.HandEyeCalibrationBase.create('Wei'),
                              ec.hand_eye.Wei)

    def _test_method(self, name, scaled=False, init_kwargs=None, **kwargs):
        if init_kwargs is None:
            init_kwargs = {}

        # create data
        motion_a, motion_b, calib = data.get_motion_data()
        if scaled:
            scale = 2.0
            motion_b.scaleTranslation_(1.0 / scale)
        else:
            scale = None

        # estimate
        method = ec.HandEyeCalibrationBase.create(name, **init_kwargs)
        method.configure(**kwargs)
        method.set_transforms(motion_a, motion_b)
        result = method.calibrate()

        # check
        self._check_result(result, calib, scale=scale)

    def _check_result(self, result, calib, scale=None):
        # check success
        if not result.success:
            self.fail(result.message)

        # check estimate
        error = calib / result.calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

        # check scale
        if scale is not None:
            if isinstance(scale, list):
                self.assertEqual(len(result.scale), len(scale))
                for s1, s2 in zip(result.scale, scale):
                    self.assertAlmostEqual(s1, s2, places=3)
            else:
                self.assertAlmostEqual(result.scale, scale, places=3)

    def test_andreff(self):
        self._test_method('Andreff')

    def test_daniilidis(self):
        self._test_method('Daniilidis')

    def test_qcqp_hm(self):
        self._test_method('MatrixQCQP')
        self._test_method('MatrixQCQPScaled', scaled=False)
        self._test_method('MatrixQCQPScaled', scaled=True)

    def test_qcqp_dq(self):
        self._test_method('DualQuaternionQCQP')
        self._test_method('DualQuaternionQCQPScaled', scaled=False)
        self._test_method('DualQuaternionQCQPScaled', scaled=True)

    def test_qcqp_dq_norm(self):
        # create data
        motion_a, motion_b, calib = data.get_motion_data(planar=True)

        # translation norm constraint
        config_kwargs = {
            't_norm': calib.translationNorm(),
            'up_approx': [0.0, 0.0, 1.0],
        }

        # estimate
        method = ec.HandEyeCalibrationBase.create('DualQuaternionQCQP')
        method.configure(**config_kwargs)
        method.set_transforms(motion_a, motion_b)
        result = method.calibrate()

        # check
        self._check_result(result, calib)

    def test_qcqp_dq_planar(self):
        # create data
        motion_a, motion_b, calib = data.get_motion_data(planar=True)

        # planes
        plane_a = Plane(normal=np.array([0.0, 0.0, 1.0]), distance=1.0)
        plane_b = plane_a.transform(calib.inverse())

        # estimate
        method = ec.HandEyeCalibrationBase.create('DualQuaternionQCQPPlanar')
        method.set_plane_transforms(plane_a.get_transform(), plane_b.get_transform())
        method.set_transforms(motion_a, motion_b)
        result = method.calibrate()

        # check
        self._check_result(result, calib)

    def test_qcqp_dq_multiscale(self):
        # create data
        motion_a, motion_b, calib = data.get_motion_data()

        # scale motions
        scale = [2.0, 4.0]
        motion_b1 = motion_b.scaleTranslation(1.0 / scale[0])
        motion_b2 = motion_b.scaleTranslation(1.0 / scale[1])

        # estimate
        method = ec.HandEyeCalibrationBase.create('DualQuaternionQCQPScaled')
        method.set_transforms([motion_a, motion_a], [motion_b1, motion_b2])
        result = method.calibrate()

        # check
        self._check_result(result, calib, scale=scale)

    def test_schmidt(self):
        for name, scaled, improved in itertools.product(['SchmidtDQ', 'SchmidtHM'], [False, True], [False, True]):
            self._test_method(name, scaled=scaled, improved=improved)

    def test_wei(self):
        self._test_method('Wei', scaled=False)
        self._test_method('Wei', scaled=True)

    def test_ransac(self):
        self._test_method('HandEyeRANSAC', init_kwargs={
            'method_name': 'DualQuaternionQCQP',
            'nreps': 10,
            'rot_thresh': math.inf,
            'trans_thresh': math.inf,
        })
