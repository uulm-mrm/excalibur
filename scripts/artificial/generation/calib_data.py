#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np

from excalibur.io.calibration import CalibrationManager
from excalibur.generation.transform import random_transforms, RandomType
import motion3d as m3d


TRANSFORM_TYPE = m3d.TransformType.kQuaternion


def main():
    parser = argparse.ArgumentParser(description="Generate calibration data.")
    parser.add_argument('base', type=str,
                        help="Base directory with poses")
    parser.add_argument('--n', type=int, default=1,
                        help="Number of random calibrations")
    parser.add_argument('--nmulti', type=int, default=1,
                        help="Number of Ys for each X")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    run(base=args.base, n_reps=args.n, n_multi=args.nmulti, seed=args.seed)


def run(base: Union[Path, str], n_reps: int = 1, n_multi: int = 1, seed: Optional[int] = None):
    # process arguments
    base_path = Path(base)

    # random seed
    if seed is not None:
        np.random.seed(seed)

    # identity transform for later
    identity = m3d.TransformInterface.Factory(TRANSFORM_TYPE)

    # iterate calibrations
    for rep_num in range(n_reps):
        # print status
        print(f"Calib data repetition {rep_num+1}/{n_reps}")

        # generate random calibration
        calib_list = random_transforms(1 + n_multi, 1.0, RandomType.UNIFORM, np.pi, RandomType.UNIFORM)
        calib_x = calib_list[0]
        calibs_y = calib_list[1:]

        calib_x_inv = calib_x.inverse().normalized_()
        calibs_y_inv = calibs_y.inverse().normalized_()

        # generate calibration manager
        manager = CalibrationManager()
        manager.add('xa', 'xb', calib_x)
        if len(calibs_y) == 1:
            manager.add('ya', 'yb', calibs_y[0])
        else:
            for ny, ty in enumerate(calibs_y):
                manager.add('ya', f'yb{ny}', ty)

        # iterate pose files
        for poses_file in base_path.glob('**/poses.m3d'):
            # print status
            print(f"Processing poses: {poses_file}")

            # load poses
            poses_a_data, status = m3d.M3DReader.read(str(poses_file))
            if status != m3d.M3DIOStatus.kSuccess:
                raise RuntimeError(f"Could not load poses: {poses_file} ({status})")
            poses_a = poses_a_data.getTransforms()

            # generate poses for b
            poses_b = poses_a.applyPre(calib_x_inv).normalized_()

            # generate motions for hand-eye calibration
            motions_a = poses_a.asMotions().normalized_()
            motions_b = motions_a.apply(calib_x_inv, calib_x).normalized_()

            # generate poses for herw calibration
            poses_b_herw_list = [poses_a.apply(calib_y_inv, calib_x).normalized_()
                                 for calib_y_inv in calibs_y_inv]

            # adjust types
            poses_a = poses_a.asType(TRANSFORM_TYPE).normalized_()
            poses_b = poses_b.asType(TRANSFORM_TYPE).normalized_()
            motions_a = motions_a.asType(TRANSFORM_TYPE).normalized_()
            motions_b = motions_b.asType(TRANSFORM_TYPE).normalized_()
            poses_b_herw_list = [poses_b_herw.asType(TRANSFORM_TYPE).normalized_()
                                 for poses_b_herw in poses_b_herw_list]

            # check data
            for pa, pb in zip(poses_a, poses_b):
                # point set registration
                diff = pa.getTranslation() - calib_x.transformPoint(pb.getTranslation())
                assert np.linalg.norm(diff) < 1e-6

                # pose set registration
                cycle = pa.inverse() * calib_x * pb
                assert cycle.translationNorm() < 1e-6 and cycle.rotationNorm() < 1e-6

            for ma, mb in zip(motions_a, motions_b):
                # hand-eye calibration
                cycle = calib_x_inv * ma.inverse() * calib_x * mb
                assert cycle.translationNorm() < 1e-6 and cycle.rotationNorm() < 1e-6

            for calib_y, poses_b_herw in zip(calibs_y, poses_b_herw_list):
                for pa, pb in zip(poses_a, poses_b_herw):
                    # hand-eye robot-world calibration
                    cycle = calib_x_inv * pa.inverse() * calib_y * pb
                    assert cycle.translationNorm() < 1e-6 and cycle.rotationNorm() < 1e-6

            # generate output paths
            rep_path = poses_file.parent / f'{rep_num:d}'
            pose_path = rep_path / 'pose'
            pose_path.mkdir(parents=True, exist_ok=True)
            he_path = rep_path / 'he'
            he_path.mkdir(parents=True, exist_ok=True)
            herw_path = rep_path / 'herw'
            herw_path.mkdir(parents=True, exist_ok=True)

            # write calibration
            manager.save(str(rep_path / 'calib.yaml'))

            # write poses
            poses_a_data = m3d.MotionData(TRANSFORM_TYPE, poses_a, origin=calib_x)
            poses_b_data = m3d.MotionData(TRANSFORM_TYPE, poses_b, origin=identity)
            m3d.M3DWriter.write(str(pose_path / 'a.m3d'), poses_a_data, m3d.M3DFileType.kBinary)
            m3d.M3DWriter.write(str(pose_path / 'b.m3d'), poses_b_data, m3d.M3DFileType.kBinary)

            # write he
            motions_a_data = m3d.MotionData(TRANSFORM_TYPE, motions_a, origin=calib_x)
            motions_b_data = m3d.MotionData(TRANSFORM_TYPE, motions_b, origin=identity)
            m3d.M3DWriter.write(str(he_path / 'a.m3d'), motions_a_data, m3d.M3DFileType.kBinary)
            m3d.M3DWriter.write(str(he_path / 'b.m3d'), motions_b_data, m3d.M3DFileType.kBinary)

            # write herw
            for herw_idx, (calib_y, poses_b_herw) in enumerate(zip(calibs_y, poses_b_herw_list)):
                # subdir
                herw_path_sub = herw_path / f'{herw_idx:d}'
                herw_path_sub.mkdir(parents=False, exist_ok=True)

                # write data
                poses_a_herw_data = m3d.MotionData(TRANSFORM_TYPE, poses_a, origin=calib_x)
                poses_b_herw_data = m3d.MotionData(TRANSFORM_TYPE, poses_b_herw, origin=calib_y)
                m3d.M3DWriter.write(str(herw_path_sub / 'a.m3d'), poses_a_herw_data, m3d.M3DFileType.kBinary)
                m3d.M3DWriter.write(str(herw_path_sub / 'b.m3d'), poses_b_herw_data, m3d.M3DFileType.kBinary)


if __name__ == '__main__':
    main()
