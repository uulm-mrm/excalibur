from pathlib import Path
import tempfile

import unittest
import yaml

from excalibur.io.calibration import CalibrationManager
from excalibur.io.transforms import store_transform_container

import data
import utils


SCRIPTS_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'calibration'


class TestCalibrationScripts(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = Path(tmpdir)
            super(TestCalibrationScripts, self).run(result)

    def _add_stamps(self, transforms):
        print(transforms)

    def _execute_script(self, path, *args):
        print(f"\n===== {path.stem} =====")
        retcode, stdout = utils.run_script(path, *args)
        print(stdout)
        if retcode != 0:
            self.fail(f"Returncode {retcode} is not 0")

    def _check_result(self, path, calib):
        # check output
        self.assertTrue(path.exists())
        manager = CalibrationManager.load(path)
        result_calib = manager.get('a', 'b')

        # check estimate
        self.assertIsNotNone(result_calib)
        error = calib / result_calib
        self.assertAlmostEqual(error.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error.translationNorm(), 0.0, places=4)

    def test_frame2frame(self):
        # create data
        poses_a, poses_b, calib = data.get_target_data()
        poses_a = data.add_stamps_to_transforms(poses_a)
        poses_b = data.add_stamps_to_transforms(poses_b)

        # store data
        filename_a = self.tmpdir / 'a.m3d'
        filename_b = self.tmpdir / 'b.m3d'
        filename_out = self.tmpdir / 'out.yaml'
        store_transform_container(filename_a, poses_a)
        store_transform_container(filename_b, poses_b)

        # generate arguments
        args = [
            SCRIPTS_PATH / 'configs' / 'frame2frame.yaml',
            '--a', filename_a,
            '--b', filename_b,
            '--novis',
            '--output', filename_out,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'frame2frame.py', *args)

        # check output
        self._check_result(filename_out, calib)

    def _test_hand_eye(self, planar: bool = False, scaled: bool = False):
        # create data
        motion_a, motion_b, calib = data.get_motion_data(planar=planar)
        motion_a.asPoses_()
        motion_b.asPoses_()
        motion_a = data.add_stamps_to_transforms(motion_a)
        motion_b = data.add_stamps_to_transforms(motion_b)

        # scaled
        if scaled:
            motion_b.scaleTranslation_(0.5)

        # store data
        filename_a = self.tmpdir / 'a.m3d'
        filename_b = self.tmpdir / 'b.m3d'
        filename_out = self.tmpdir / 'out.yaml'
        store_transform_container(filename_a, motion_a)
        store_transform_container(filename_b, motion_b)

        # config
        if scaled:
            cfg_file = SCRIPTS_PATH / 'configs' / 'hand_eye_scaled.yaml'
        else:
            cfg_file = SCRIPTS_PATH / 'configs' / 'hand_eye.yaml'

        # generate arguments
        args = [
            cfg_file,
            '--a', filename_a,
            '--b', filename_b,
            '--novis',
            '--output', filename_out,
        ]

        # planar config
        if planar:
            norm_cfg_file = self.tmpdir / 'hand_eye_norm.yaml'
            with open(norm_cfg_file, 'w') as norm_cfg:
                yaml.dump({
                    't_norm': calib.translationNorm(),
                    'up_approx': [0.0, 0.0, 1.0],
                }, norm_cfg)
            args.extend(['--norm-cfg', norm_cfg_file])

        # runs script
        self._execute_script(SCRIPTS_PATH / 'hand_eye.py', *args)

        # check output
        self._check_result(filename_out, calib)

    def test_hand_eye(self):
        self._test_hand_eye(planar=False, scaled=False)
        self._test_hand_eye(planar=True, scaled=False)
        self._test_hand_eye(planar=False, scaled=True)

    def _test_herw(self, planar: bool = False, y_norm: bool = False):
        # create data
        poses_a, poses_b, calib_x, calib_y = data.get_herw_pose_data(planar=planar)
        poses_a = data.add_stamps_to_transforms(poses_a)
        poses_b = data.add_stamps_to_transforms(poses_b)

        # store data
        filename_a = self.tmpdir / 'a.m3d'
        filename_b = self.tmpdir / 'b.m3d'
        filename_out = self.tmpdir / 'out.yaml'
        store_transform_container(filename_a, poses_a)
        store_transform_container(filename_b, poses_b)

        # generate arguments
        args = [
            SCRIPTS_PATH / 'configs' / 'herw.yaml',
            '--a', filename_a,
            '--b', filename_b,
            '--novis',
            '--output', filename_out,
        ]

        # planar config
        if planar:
            norm_cfg_file = self.tmpdir / 'herw_norm.yaml'
            with open(norm_cfg_file, 'w') as norm_cfg:
                if y_norm:
                    norm_cfg_data = {
                        't_norm_y': calib_y.translationNorm(),
                        'up_approx': [0.0, 0.0, 1.0],
                    }
                else:
                    norm_cfg_data = {
                        't_norm_x': calib_x.translationNorm(),
                        'up_approx': [0.0, 0.0, 1.0],
                    }
                yaml.dump(norm_cfg_data, norm_cfg)
            args.extend(['--norm-cfg', norm_cfg_file])

        # runs script
        self._execute_script(SCRIPTS_PATH / 'herw.py', *args)

        # check output
        self.assertTrue(filename_out.exists())
        manager = CalibrationManager.load(filename_out)
        result_calib_x = manager.get('x_from', 'x_to')
        result_calib_y = manager.get('y_from', 'y_to')

        # check estimate
        self.assertIsNotNone(result_calib_x)
        self.assertIsNotNone(result_calib_y)
        error_x = calib_x / result_calib_x
        error_y = calib_y / result_calib_y
        self.assertAlmostEqual(error_x.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error_x.translationNorm(), 0.0, places=3)
        self.assertAlmostEqual(error_y.rotationNorm(), 0.0, places=4)
        self.assertAlmostEqual(error_y.translationNorm(), 0.0, places=3)

    def test_herw(self):
        self._test_herw(planar=False)
        self._test_herw(planar=True, y_norm=False)
        self._test_herw(planar=True, y_norm=True)

    def test_point2point(self):
        # create data
        poses_a, poses_b, calib = data.get_target_data()
        poses_a = data.add_stamps_to_transforms(poses_a)
        poses_b = data.add_stamps_to_transforms(poses_b)

        # store data
        filename_a = self.tmpdir / 'a.m3d'
        filename_b = self.tmpdir / 'b.m3d'
        filename_out = self.tmpdir / 'out.yaml'
        store_transform_container(filename_a, poses_a)
        store_transform_container(filename_b, poses_b)

        # generate arguments
        args = [
            SCRIPTS_PATH / 'configs' / 'point2point.yaml',
            '--a', filename_a,
            '--b', filename_b,
            '--novis',
            '--output', filename_out,
        ]

        # runs script
        self._execute_script(SCRIPTS_PATH / 'point2point.py', *args)

        # check output
        self._check_result(filename_out, calib)
