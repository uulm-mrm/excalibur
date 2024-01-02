#!/usr/bin/env python3
import argparse
import signal
import sys

import open3d as o3d

from excalibur.io.calibration import CalibrationManager
from excalibur.ros.pointcloud2 import msg_to_numpy
from excalibur.ros.reader import Reader
from excalibur.visualization.cloud import cloud_to_open3d
from excalibur.visualization.colors import get_color_tuple


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Show two point clouds transformed into the same coordinate frame.")
    parser.add_argument('rosbag', type=str, help="rosbag directory")
    parser.add_argument('topic1', type=str, help="point cloud topic of lidar 1")
    parser.add_argument('topic2', type=str, help="point cloud topic of lidar 2")
    parser.add_argument('calib_file', type=str, help="calibration file (*.yaml)")
    parser.add_argument('frame1', type=str, help="frame identifier of lidar 1")
    parser.add_argument('frame2', type=str, help="frame identifier of lidar 2")
    parser.add_argument('--dt', type=float, default=None, help="time step [s]")
    parser.add_argument('--queue', type=int, default=20, help="queue size for topic synchronization")
    parser.add_argument('--slop', type=float, default=0.1, help="maximum slop for topic synchronization [s]")
    args = parser.parse_args()

    # process input arguments
    if args.dt is not None:
        args.dt = int(args.dt * 1e9)
    if args.slop is not None:
        args.slop = int(args.slop * 1e9)

    # load calibration
    manager = CalibrationManager.load(args.calib_file)
    transform = manager.get(args.frame1, args.frame2)
    if transform is None:
        print(f"error: transformation {args.frame1}->{args.frame2} not found in '{args.calib_file}'")
        sys.exit(-1)

    # open reader
    with Reader(args.rosbag) as reader:

        # config
        reader.set_dt(args.dt)
        reader.set_verbose(True)

        # prepare topics
        topics = [args.topic1, args.topic2]

        # iterate messages
        for messages in reader.messages_sync(topics=topics, queue_size=args.queue, slop=args.slop):
            # convert clouds to numpy
            cloud1 = msg_to_numpy(messages[args.topic1].msg, ('x', 'y', 'z'))
            cloud2 = msg_to_numpy(messages[args.topic2].msg, ('x', 'y', 'z'))

            # transform
            cloud2 = cloud2.reshape((cloud2.shape[0] * cloud2.shape[1], cloud2.shape[2]))[:, :3]
            cloud2 = transform.transformCloud(cloud2.T).T

            # visualize
            cloud1 = cloud_to_open3d(cloud1.copy(), colors=get_color_tuple('tab:blue'))
            cloud2 = cloud_to_open3d(cloud2.copy(), colors=get_color_tuple('tab:orange'))
            o3d.visualization.draw_geometries([cloud1, cloud2])


if __name__ == '__main__':
    main()
