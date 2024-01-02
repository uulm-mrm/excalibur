#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from excalibur.io.cloud import StampedCloudIterator
from excalibur.targets.lidar.plane import detect_plane, LidarPlaneConfig
from excalibur.visualization.cloud import crop_cloud, visualize_cloud


def rosbag_generator(args):
    # import ROS dependencies only when required
    from excalibur.ros.reader import Reader
    from excalibur.ros.pointcloud2 import iterate_pointcloud2_data

    # open reader
    with Reader(args.path) as reader:

        # config
        reader.set_dt(args.dt)
        reader.set_verbose(True)

        # iterate images
        for stamp, cloud in iterate_pointcloud2_data(reader, args.topic, fields=('x', 'y', 'z')):
            yield stamp, cloud


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Estimate ground plane in lidar point clouds.")
    parser.add_argument('path', type=str, help="rosbag or data export directory")
    parser.add_argument('config', type=str, help="detector configuration")
    parser.add_argument('--topic', type=str, nargs='?', help="cloud topic for rosbag processing")
    parser.add_argument('--output', type=str, default=None, help="output file (*.m3d)")
    parser.add_argument('--first', action='store_true', help="store first detection without asking")
    parser.add_argument('--dt', type=float, default=None, help="time step [s]")
    parser.add_argument('--seed', type=int, default=None, help="random seed")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    # process input arguments
    if args.dt is not None:
        args.dt = int(args.dt * 1e9)

    if args.seed is not None:
        np.random.seed(args.seed)

    # detection config
    cfg = LidarPlaneConfig.from_yaml(args.config)

    # initialize output
    output_plane = None

    # initialize generator
    if args.topic is None:
        iterator = StampedCloudIterator(args.path)
        if iterator.starting_time is None:
            print(f"No clouds found in '{args.path}'")
            return
        iterator.set_dt(args.dt)
        iterator.set_verbose(True)
        data_generator = iterator.iter()
    else:
        data_generator = rosbag_generator(args)

    # iterate point clouds
    for stamp, cloud in data_generator:
        # estimate plane
        plane = detect_plane(cloud, cfg, debug=args.debug)
        if plane is None:
            continue

        # print and visualize
        print(f"Plane | Normal=[{plane.normal[0]:.4f}, {plane.normal[1]:.4f}, {plane.normal[2]:.4f}], "
              f"distance={plane.distance:.4f}")
        if args.show or args.debug:
            cloud_cropped = crop_cloud(cloud, min_range=cfg.min_dist, max_range=cfg.max_dist)
            visualize_cloud(cloud_cropped, pose=plane.get_transform())

        # check storage
        if args.output is not None:
            if not args.first:
                key = input("Enter S to store or just press Enter to continue...")
                if key.lower() != 's':
                    continue
            output_plane = plane
            break

    # save
    if args.output is not None and output_plane is not None:
        print(f"\nSave plane to '{args.output}'")

        # create output dir
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # save plane
        output_plane.save(args.output)


if __name__ == '__main__':
    main()
