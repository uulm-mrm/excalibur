#!/usr/bin/env python3
import argparse
from pathlib import Path

import calib_data
import noise_data

import surface_poses


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Generate calibration data.")
    parser.add_argument('noise_levels', type=str,
                        help="Noise levels config (*.yaml)")
    parser.add_argument('output', type=str,
                        help="Output directory")
    parser.add_argument('--ncalib', type=int, default=1,
                        help="Number of random calibrations")
    parser.add_argument('--nmulti', type=int, default=1,
                        help="Number of Ys for each X")
    parser.add_argument('--nnoise', type=int, default=1,
                        help="Number of random noise collections")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    # process arguments
    output = Path(args.output)
    data_output = output / 'data'
    noise_output = output / 'noise'
    calib_seed = args.seed
    noise_seed = args.seed + 1 if args.seed is not None else None

    # poses
    surface_poses.run(output=data_output, multi_scale=True)

    # calib
    calib_data.run(base=data_output, n_reps=args.ncalib, n_multi=args.nmulti, seed=calib_seed)

    # noise
    noise_poses = sorted(data_output.glob('**/poses.m3d'))[0]
    noise_data.run(poses_file=noise_poses, noise_levels_file=args.noise_levels, output=noise_output,
                   n_reps=args.nnoise, n_multi=args.nmulti, seed=noise_seed)


if __name__ == '__main__':
    main()
