from pathlib import Path
import tempfile

import motion3d as m3d
import unittest

from excalibur.io.calibration import CalibrationManager
from excalibur.io.transforms import store_transform_container

from . import data
from . import utils


SCRIPTS_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'utils'


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

    def _check_output(self, path, ref_transforms, ttype=None):
        data, status = m3d.M3DReader.read(str(path))
        self.assertEqual(status, m3d.M3DIOStatus.kSuccess)
        if ttype is not None:
            self.assertEqual(data.getTransformType(), ttype)

        transforms = data.getTransforms()
        self.assertEqual(transforms.hasPoses(), ref_transforms.hasPoses())
        self.assertEqual(transforms.hasStamps(), ref_transforms.hasStamps())
        self.assertEqual(len(transforms), len(ref_transforms))
        for t, t_ref in zip(transforms, ref_transforms):
            diff = t / t_ref
            self.assertAlmostEqual(diff.translationNorm(), 0.0, 6)
            self.assertAlmostEqual(diff.rotationNorm(), 0.0, 6)

    def test_m3d_apply(self):
        script_path = SCRIPTS_PATH / 'm3d_apply.py'

        # generate input data
        poses, _, calib = data.get_target_data()
        poses.asType_(m3d.TransformType.kMatrix)
        input_file = self.tmpdir / 'input.m3d'
        store_transform_container(input_file, poses)

        manager = CalibrationManager()
        manager.add('a', 'b', calib)
        calib_file = self.tmpdir / 'calib.yaml'
        manager.save(calib_file)

        # invert
        output_file_inverse = self.tmpdir / 'inverse.m3d'
        args_inverse = [input_file, output_file_inverse, '--inverse']
        self._execute_script(script_path, *args_inverse)
        self._check_output(output_file_inverse, poses.inverse(), m3d.TransformType.kMatrix)

        # change
        output_file_change = self.tmpdir / 'change.m3d'
        args_change = [input_file, output_file_change, '--change', calib_file, 'a', 'b']
        self._execute_script(script_path, *args_change)
        self._check_output(output_file_change, poses.changeFrame(calib), m3d.TransformType.kMatrix)

        # transform type
        output_file_ttype = self.tmpdir / 'ttype.m3d'
        args_ttype = [input_file, output_file_ttype, '--type', 'kEuler']
        self._execute_script(script_path, *args_ttype)
        self._check_output(output_file_ttype, poses, m3d.TransformType.kEuler)

        # ascii
        output_file_ascii = self.tmpdir / 'ascii.m3d'
        args_ascii = [input_file, output_file_ascii, '--ascii']
        self._execute_script(script_path, *args_ascii)
        self._check_output(output_file_ascii, poses, m3d.TransformType.kMatrix)
