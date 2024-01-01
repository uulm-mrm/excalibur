#!/usr/bin/env python3
import argparse
from enum import auto, Enum
import sys

import motion3d as m3d
import numpy as np

from excalibur.io.cloud import StampedCloudIterator
from excalibur.io.transforms import store_transform_container
from excalibur.targets.lidar.board import detect_board, LidarBoardConfig
from excalibur.targets.lidar.sphere import detect_sphere, LidarSphereConfig
from excalibur.utils.parsing import ParseEnum
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

        # iterate clouds
        for stamp, cloud in iterate_pointcloud2_data(reader, args.topic, fields=('x', 'y', 'z')):
            yield stamp, cloud


class TargetType(Enum):
    BOARD = auto()
    SPHERE = auto()


def detect_sphere_pose(cloud, cfg):
    sphere = detect_sphere(cloud, cfg)
    if sphere is None:
        return None
    return m3d.EulerTransform(sphere.center, [0.0, 0.0, 0.0])


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Detect lidar calibration targets.")
    parser.add_argument('path', type=str, help="rosbag or data export directory")
    parser.add_argument('type', action=ParseEnum, enum_type=TargetType, help="target type")
    parser.add_argument('config', type=str, help="detector configuration")
    parser.add_argument('--topic', type=str, nargs='?', help="cloud topic for rosbag processing")
    parser.add_argument('--output', type=str, default=None, help="output file (*.m3d)")
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
    try:
        if args.type == TargetType.BOARD:
            cfg = LidarBoardConfig.from_yaml(args.config)

            def detect_target(_cloud):
                return detect_board(_cloud, cfg, debug=args.debug)

        elif args.type == TargetType.SPHERE:
            cfg = LidarSphereConfig.from_yaml(args.config)

            def detect_target(_cloud):
                return detect_sphere_pose(_cloud, cfg)

        else:
            print(f"Error: unsupported target type: '{args.type}'")
            sys.exit(-1)
    except FileNotFoundError:
        print(f"Error: Could not load '{args.type.name}' configuration: '{args.config}'")
        sys.exit(-1)

    # initialize container
    container = m3d.TransformContainer(has_poses=True, has_stamps=True)

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
        # estimate target pose
        target_pose = detect_target(cloud)
        if target_pose is None:
            continue

        # save
        container.append(stamp, target_pose)
        print(f"Detected Targets: {len(container)}")

        # visualize
        if args.show or args.debug:
            cloud_cropped = crop_cloud(cloud, min_range=cfg.min_range, max_range=cfg.max_range)
            visualize_cloud(cloud_cropped, pose=target_pose)

    # save
    if args.output is not None:
        print(f"\nExport {len(container)} detections to '{args.output}'")
        store_transform_container(args.output, container)


if __name__ == '__main__':
    main()
