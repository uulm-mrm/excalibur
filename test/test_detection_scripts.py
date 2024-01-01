from pathlib import Path
import tempfile

import motion3d as m3d
import unittest

import utils

from excalibur.fitting.plane import Plane


SCRIPTS_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'detection'
DATA_PATH = Path(__file__).resolve().parent / 'detection'
CONFIGS_PATH = DATA_PATH / 'configs'


class TestCalibrationScripts(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = Path(tmpdir)
            super(TestCalibrationScripts, self).run(result)

    def _execute_script(self, path, *args):
        print(f"\n===== {path.stem} =====")
        retcode, stdout = utils.run_script(path, *args)
        print(stdout)
        if retcode != 0:
            self.fail(f"Returncode {retcode} is not 0")

    def _check_output(self, path):
        data, status = m3d.M3DReader.read(str(path))
        self.assertEqual(status, m3d.M3DIOStatus.kSuccess)

        transforms = data.getTransforms()
        self.assertTrue(transforms.hasPoses())
        self.assertTrue(transforms.hasStamps())
        self.assertEqual(len(transforms), 1)
        self.assertEqual(transforms.stamp_at(0).toSec(), 1.5)
        return transforms[0].asType(m3d.TransformType.kMatrix)

    def test_camera_charuco(self):
        data_path = DATA_PATH / 'camera_charuco'
        output_file = self.tmpdir / 'out.m3d'

        # generate arguments
        args = [
            data_path,
            'CHARUCO',
            CONFIGS_PATH / 'camera_charuco.yaml',
            '--output', output_file,
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'camera_target.py', *args)

        # check result
        transform = self._check_output(output_file)
        translation = transform.getTranslation()
        rotmat = transform.getRotationMatrix()
        self.assertLess(translation[0], 1.0)  # x pos < 1.0
        self.assertLess(translation[1], 0.0)  # y pos < 0.0
        self.assertGreater(translation[2], 1.0)  # z pos > 1.0
        self.assertGreater(rotmat[:, 0].dot([1.0, 0.0, 0.0]), 0.0)  # x-axis right
        self.assertGreater(rotmat[:, 1].dot([0.0, -1.0, 0.0]), 0.0)  # y-axis top
        self.assertGreater(rotmat[:, 2].dot([0.0, 0.0, -1.0]), 0.0)  # z-axis to cam

    def test_camera_checkerboard(self):
        data_path = DATA_PATH / 'camera_checkerboard'
        output_file = self.tmpdir / 'out.m3d'

        # generate arguments
        args = [
            data_path,
            'CHECKERBOARD',
            CONFIGS_PATH / 'camera_checkerboard.yaml',
            '--output', output_file,
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'camera_target.py', *args)

        # check result
        transform = self._check_output(output_file)
        translation = transform.getTranslation()
        rotmat = transform.getRotationMatrix()
        self.assertLess(translation[0], 1.0)  # x pos < 1.0
        self.assertLess(translation[1], 0.0)  # y pos < 0.0
        self.assertGreater(translation[2], 1.0)  # z pos > 1.0
        self.assertGreater(rotmat[:, 0].dot([1.0, 0.0, 0.0]), 0.0)  # x-axis right
        self.assertGreater(rotmat[:, 1].dot([0.0, -1.0, 0.0]), 0.0)  # y-axis top
        self.assertGreater(rotmat[:, 2].dot([0.0, 0.0, -1.0]), 0.0)  # z-axis to cam

    def test_camera_checkerboard_combi(self):
        data_path = DATA_PATH / 'camera_checkerboard'
        output_file = self.tmpdir / 'out.m3d'

        # generate arguments
        args = [
            data_path,
            'CB_COMBI',
            CONFIGS_PATH / 'camera_checkerboard_combi.yaml',
            '--output', output_file,
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'camera_target.py', *args)

        # check result
        transform = self._check_output(output_file)
        translation = transform.getTranslation()
        rotmat = transform.getRotationMatrix()
        self.assertLess(translation[0], 1.0)  # x pos < 1.0
        self.assertLess(translation[1], 0.0)  # y pos < 0.0
        self.assertGreater(translation[2], 1.0)  # z pos > 1.0
        self.assertGreater(rotmat[:, 0].dot([1.0, 0.0, 0.0]), 0.0)  # x-axis right
        self.assertGreater(rotmat[:, 1].dot([0.0, -1.0, 0.0]), 0.0)  # y-axis top
        self.assertGreater(rotmat[:, 2].dot([0.0, 0.0, -1.0]), 0.0)  # z-axis to cam

    def test_lidar_board(self):
        data_path = DATA_PATH / 'lidar_board'
        output_file = self.tmpdir / 'out.m3d'

        # generate arguments
        args = [
            data_path,
            'BOARD',
            CONFIGS_PATH / 'lidar_board.yaml',
            '--output', output_file,
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'lidar_target.py', *args)

        # check result
        transform = self._check_output(output_file)
        translation = transform.getTranslation()
        rotmat = transform.getRotationMatrix()
        self.assertGreater(translation[0], 1.0)  # x pos > 1.0
        self.assertGreater(translation[0], 1.0)  # y pos > 1.0
        self.assertGreater(translation[2], 0.0)  # z pos > 0.0
        self.assertGreater(rotmat[:, 0].dot([0.0, -1.0, 0.0]), 0.0)  # x-axis right
        self.assertGreater(rotmat[:, 1].dot([0.0, 0.0, 1.0]), 0.0)  # y-axis top
        self.assertGreater(rotmat[:, 2].dot([-1.0, 0.0, 0.0]), 0.0)  # z-axis to lidar

    def test_lidar_plane(self):
        data_path = DATA_PATH / 'lidar_sphere'
        output_file = self.tmpdir / 'out.yaml'

        # generate arguments
        args = [
            data_path,
            CONFIGS_PATH / 'lidar_plane.yaml',
            '--output', output_file,
            '--first',
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'lidar_plane.py', *args)

        # check result
        plane = Plane.load(output_file)
        self.assertGreater(plane.normal.dot([0.0, 0.0, 1.0]), 0.5)  # normal up
        self.assertLess(plane.distance, -1.0)  # plane under lidar

    def test_lidar_sphere(self):
        data_path = DATA_PATH / 'lidar_sphere'
        output_file = self.tmpdir / 'out.m3d'

        # generate arguments
        args = [
            data_path,
            'SPHERE',
            CONFIGS_PATH / 'lidar_sphere.yaml',
            '--output', output_file,
            '--seed', 0,
        ]

        # run script
        self._execute_script(SCRIPTS_PATH / 'lidar_target.py', *args)

        # check result
        transform = self._check_output(output_file)
        translation = transform.getTranslation()
        self.assertGreater(translation[0], 1.0)  # x pos > 1.0
        self.assertGreater(translation[0], 1.0)  # y pos > 1.0
        self.assertGreater(translation[2], 0.0)  # z pos > 0.0
