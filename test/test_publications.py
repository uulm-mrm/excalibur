from pathlib import Path

import unittest

import utils


PUBLICATIONS_PATH = Path(__file__).resolve().parent.parent / 'publications'


class TestPublications(unittest.TestCase):
    def _execute_script(self, path):
        print(f"\n===== {path.stem} =====")
        retcode, stdout = utils.run_script(path)
        print(stdout)
        if retcode != 0:
            self.fail(f"Returncode {retcode} is not 0")

    def test_hand_eye(self):
        base_path = PUBLICATIONS_PATH / 'hand_eye'
        self._execute_script(base_path / 'brookshire.py')
        self._execute_script(base_path / 'euroc.py')
        self._execute_script(base_path / 'kitti_odometry.py')
        self._execute_script(base_path / 'kitti_odometry_planar.py')

    def test_hand_eye_monocular(self):
        base_path = PUBLICATIONS_PATH / 'hand_eye_monocular'
        self._execute_script(base_path / 'brookshire.py')
        self._execute_script(base_path / 'euroc.py')
        self._execute_script(base_path / 'euroc_multiscale.py')

    def test_herw_infrastructure(self):
        base_path = PUBLICATIONS_PATH / 'herw_infrastructure'
        self._execute_script(base_path / 'ali_herw.py')
        self._execute_script(base_path / 'lehr_infrastructure.py')
