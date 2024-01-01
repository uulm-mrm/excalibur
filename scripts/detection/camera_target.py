#!/usr/bin/env python3
import argparse
from enum import auto, Enum
import signal
import sys

import cv2
import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np

from excalibur.io.image import StampedImageIterator
from excalibur.io.transforms import store_transform_container
from excalibur.targets.camera.charuco import CharucoBoardConfig, detect_charuco
from excalibur.targets.camera.checkerboard import CheckerboardConfig, detect_checkerboard
from excalibur.targets.camera.checkerboard_combi import CheckerboardCombiConfig, detect_checherboard_combi
from excalibur.utils.parsing import ParseEnum
from excalibur.visualization.image import draw_frame_axes


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def rosbag_generator(args):
    # import ROS dependencies only when required
    from excalibur.ros.reader import Reader
    from excalibur.ros.image import iterate_image_data as iterate_image_data_ros

    # open reader
    with Reader(args.path) as reader:

        # config
        reader.set_dt(args.dt)
        reader.set_verbose(True)

        # iterate images
        for stamp, img, intrinsics in iterate_image_data_ros(reader, args.topic, desired_encoding='RGB8'):
            yield stamp, img, intrinsics


class TargetType(Enum):
    CB_COMBI = auto()
    CHARUCO = auto()
    CHECKERBOARD = auto()


def detect_cb_combi_pose(img, cfg, intrinsics, debug):
    # grayscale for detector
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # estimate board pose
    board_det, aruco_dets = detect_checherboard_combi(img_gray, cfg, intrinsics=intrinsics, debug=debug)
    if board_det is None:
        return None
    return board_det.pose


def detect_charuco_pose(img, cfg, intrinsics, debug):
    # grayscale for detector
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # estimate board pose
    detection = detect_charuco(img_gray, cfg, intrinsics=intrinsics, debug=debug)
    if detection is None:
        return None
    return detection.pose


def detect_checkerboard_pose(img, cfg, intrinsics, _debug):
    # grayscale for detector
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # estimate board pose
    detection = detect_checkerboard(img_gray, cfg, intrinsics=intrinsics)
    if detection is None:
        return None
    return detection.pose


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Detect camera calibration targets.")
    parser.add_argument('path', type=str, help="rosbag or data export directory")
    parser.add_argument('type', action=ParseEnum, enum_type=TargetType, help="target type")
    parser.add_argument('config', type=str, help="detector configuration")
    parser.add_argument('--topic', type=str, nargs='?', help="image topic for rosbag processing")
    parser.add_argument('--flip', action='store_true', help="flip image before detection")
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
        if args.type == TargetType.CB_COMBI:
            cfg = CheckerboardCombiConfig.from_yaml(args.config)

            def detect_target(_img, _intrinsics):
                return detect_cb_combi_pose(_img, cfg, _intrinsics, args.debug)

        elif args.type == TargetType.CHARUCO:
            cfg = CharucoBoardConfig.from_yaml(args.config)

            def detect_target(_img, _intrinsics):
                return detect_charuco_pose(_img, cfg, _intrinsics, args.debug)

        elif args.type == TargetType.CHECKERBOARD:
            cfg = CheckerboardConfig.from_yaml(args.config)

            def detect_target(_img, _intrinsics):
                return detect_checkerboard_pose(_img, cfg, _intrinsics, args.debug)

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
        iterator = StampedImageIterator(args.path)
        if iterator.starting_time is None:
            print(f"No images found in '{args.path}'")
            return
        iterator.set_dt(args.dt)
        iterator.set_verbose(True)
        data_generator = iterator.iter(desired_encoding='RGB8')
    else:
        data_generator = rosbag_generator(args)

    # iterate generator
    for stamp, img, intrinsics in data_generator:
        # flip
        if args.flip:
            img = cv2.flip(img, -1)

        # estimate target pose
        target_pose = detect_target(img, intrinsics)
        if target_pose is None:
            continue

        # save
        container.append(stamp, target_pose)
        print(f"Detected Targets: {len(container)}")

        # visualize
        if args.show or args.debug:
            pose_img = img.copy()
            draw_frame_axes(pose_img, [target_pose], 0.5, intrinsics)
            plt.figure()
            plt.imshow(pose_img)
            plt.show()

    # save
    if args.output is not None:
        print(f"\nExport {len(container)} detections to '{args.output}'")
        store_transform_container(args.output, container)


if __name__ == '__main__':
    main()
