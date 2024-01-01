#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import motion3d as m3d
import numpy as np
import yaml

from excalibur.generation.transform import random_transforms, RandomType
from excalibur.io.utils import load_yaml


TRANSFORM_TYPE = m3d.TransformType.kQuaternion


@dataclass
class NoiseLevel:
    trans_rel: Optional[float] = None
    rot_rel: Optional[float] = None
    trans_abs: Optional[float] = None
    rot_abs: Optional[float] = None
    rot_abs_deg: Optional[float] = None
    desc: Optional[str] = None

    def complete(self, avg_trans: float, avg_rot: float):
        if self.trans_abs is not None and (self.rot_abs is not None or self.rot_abs_deg is not None):
            # deg to rad
            if self.rot_abs is None:
                self.rot_abs = float(np.deg2rad(self.rot_abs_deg))
                new_desc = f'abs_{self.trans_abs:.4f}_{self.rot_abs_deg:.4f}deg'
            else:
                new_desc = f'abs_{self.trans_abs:.4f}_{self.rot_abs:.4f}rad'

            if self.desc is None:
                self.desc = new_desc

            # relative noise from absolute noise
            self.trans_abs = self.trans_abs
            self.rot_abs = self.rot_abs
            self.trans_rel = self.trans_abs / avg_trans * 100.0
            self.rot_rel = self.rot_abs / avg_rot * 100.0

        elif self.trans_rel is not None and self.rot_rel is not None:
            # absolute noise from relative noise
            self.trans_rel = self.trans_rel
            self.rot_rel = self.rot_rel
            self.trans_abs = avg_trans * self.trans_rel / 100.0
            self.rot_abs = avg_rot * self.rot_rel / 100.0

            if self.desc is None:
                self.desc = f'rel_{self.trans_rel:.2f}_{self.rot_rel:.2f}'

        else:
            raise RuntimeError("Either absolute or relative noise is required.")


def load_noise_levels(filename: str):
    cfg = load_yaml(filename)
    return [NoiseLevel(**noise_cfg) for noise_cfg in cfg]


def main():
    parser = argparse.ArgumentParser(description="Generate noise data.")
    parser.add_argument('poses', type=str,
                        help="File with poses for average rotation and translation")
    parser.add_argument('noise_levels', type=str,
                        help="Noise levels config (*.yaml)")
    parser.add_argument('output', type=str,
                        help="Output directory for noise data")
    parser.add_argument('--n', type=int, default=1,
                        help="Number of random noise collections")
    parser.add_argument('--nmulti', type=int, default=1,
                        help="Factor for multi-calibration")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    run(args.poses, args.noise_levels, args.output, args.n, args.nmulti, args.seed)


def run(poses_file: Union[Path, str], noise_levels_file: Union[Path, str], output: Union[Path, str],
        n_reps: int = 1, n_multi: int = 1, seed: Optional[int] = None):
    # process arguments
    output_path = Path(output)

    # load poses
    poses_data, status = m3d.M3DReader.read(str(poses_file))
    if status != m3d.M3DIOStatus.kSuccess:
        raise RuntimeError(f"Could not load poses: {poses_file} ({status})")
    poses = poses_data.getTransforms()

    # get average translation and rotation
    motions = poses.asMotions()
    avg_trans = np.mean([m.translationNorm() for m in motions])
    avg_rot = np.mean([m.rotationNorm() for m in motions])

    # load noise levels
    noise_levels = load_noise_levels(str(noise_levels_file))

    # iterate noise levels
    for noise_lvl in noise_levels:
        # print status
        print(f"Noise level: {noise_lvl}")

        # reset seed
        if seed is not None:
            np.random.seed(seed)

        # complete noise
        noise_lvl.complete(float(avg_trans), float(avg_rot))

        # output path
        noise_path_base = output_path / noise_lvl.desc
        noise_path_base.mkdir(parents=True, exist_ok=True)

        # store config
        config_data = {
            'trans_noise_rel': noise_lvl.trans_rel,
            'trans_noise_abs': noise_lvl.trans_abs,
            'rot_noise_rel': noise_lvl.rot_rel,
            'rot_noise_abs': noise_lvl.rot_abs,
            'rot_noise_abs_deg': float(np.rad2deg(noise_lvl.rot_abs)),
            'noise_desc': noise_lvl.desc,
            'seed': seed,
        }
        config_path = noise_path_base / 'config.yaml'
        with open(config_path, 'w') as file:
            yaml.dump(config_data, file)

        # repeat
        for rep_num in range(n_reps):
            # print status
            print(f"Noise repetition {rep_num + 1}/{n_reps}")

            # generate noise transforms
            noise_a = random_transforms(
                len(poses) * n_multi, noise_lvl.trans_abs, RandomType.NORMAL, noise_lvl.rot_abs, RandomType.NORMAL)
            noise_b = random_transforms(
                len(poses) * n_multi, noise_lvl.trans_abs, RandomType.NORMAL, noise_lvl.rot_abs, RandomType.NORMAL)

            # adjust types
            noise_a = noise_a.asType(TRANSFORM_TYPE).normalized_()
            noise_b = noise_b.asType(TRANSFORM_TYPE).normalized_()

            # output path
            noise_path_rep = noise_path_base / f'{rep_num:d}'
            noise_path_rep.mkdir(parents=True, exist_ok=True)

            # write data
            noise_a_data = m3d.MotionData(TRANSFORM_TYPE, noise_a)
            noise_b_data = m3d.MotionData(TRANSFORM_TYPE, noise_b)
            m3d.M3DWriter.write(str(noise_path_rep / 'a.m3d'), noise_a_data, m3d.M3DFileType.kBinary)
            m3d.M3DWriter.write(str(noise_path_rep / 'b.m3d'), noise_b_data, m3d.M3DFileType.kBinary)


if __name__ == '__main__':
    main()
