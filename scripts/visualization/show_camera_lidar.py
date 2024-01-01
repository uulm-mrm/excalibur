#!/usr/bin/env python3
import argparse
import signal
import sys

import cv2
import matplotlib.pyplot as plt

from excalibur.io.calibration import CalibrationManager
from excalibur.ros.cv_bridge import imgmsg_to_cv2
from excalibur.ros.image import camera_info_msg_to_intrinsics, get_camera_topics_info
from excalibur.ros.pointcloud2 import msg_to_numpy
from excalibur.ros.reader import Reader
from excalibur.visualization.image import draw_point_cloud


signal.signal(signal.SIGINT, signal.SIG_DFL)  # abort even when the matplotlib figure is not in focus


def main():
    # arguments
    parser = argparse.ArgumentParser(description="Project lidar point clouds on camera images.")
    parser.add_argument('rosbag', type=str, help="rosbag directory")
    parser.add_argument('img_topic', type=str, help="image topic of camera")
    parser.add_argument('lid_topic', type=str, help="point cloud topic of lidar")
    parser.add_argument('calib_file', type=str, help="calibration file (*.yaml)")
    parser.add_argument('cam_frame', type=str, help="frame identifier of camera")
    parser.add_argument('lid_frame', type=str, help="frame identifier of lidar")
    parser.add_argument('--dt', type=float, default=None, help="time step [s]")
    parser.add_argument('--queue', type=int, default=20, help="queue size for topic synchronization")
    parser.add_argument('--slop', type=float, default=0.1, help="maximum slop for topic synchronization [s]")
    parser.add_argument('--no-heq', action='store_true', help="skip histrogram equalization")
    parser.add_argument('--z-min', type=float, default=0.1, help="minimum camera distance")
    parser.add_argument('--z-max', type=float, default=100.0, help="maximum camera distance")
    parser.add_argument('--radius', type=int, default=1, help="point radius [px]")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha value for overlay")
    args = parser.parse_args()

    # process input arguments
    if args.dt is not None:
        args.dt = int(args.dt * 1e9)
    if args.slop is not None:
        args.slop = int(args.slop * 1e9)

    # load calibration
    manager = CalibrationManager.load(args.calib_file)
    transform = manager.get(args.cam_frame, args.lid_frame)
    if transform is None:
        print(f"error: transformation {args.cam_frame}->{args.lid_frame} not found in '{args.calib_file}'")
        sys.exit(-1)

    # open reader
    with Reader(args.rosbag) as reader:

        # config
        reader.set_dt(args.dt)
        reader.set_verbose(True)

        # prepare topics
        camera_topics_info = get_camera_topics_info(args.img_topic)
        topics = [camera_topics_info.camera_info_topic, camera_topics_info.image_topic, args.lid_topic]

        # iterate messages
        for messages in reader.messages_sync(topics=topics, queue_size=args.queue, slop=args.slop):
            # convert intrinsics
            camera_info_msg = messages[camera_topics_info.camera_info_topic].msg
            intrinsics = camera_info_msg_to_intrinsics(camera_info_msg, camera_topics_info.is_rect)

            # convert image
            img_msg = messages[camera_topics_info.image_topic].msg
            img = imgmsg_to_cv2(img_msg, desired_encoding='RGB8')

            # convert cloud
            cloud_msg = messages[args.lid_topic].msg
            cloud = msg_to_numpy(cloud_msg, ('x', 'y', 'z'))

            # transform cloud
            cloud = cloud.reshape((cloud.shape[0] * cloud.shape[1], cloud.shape[2]))[:, :3]
            cloud = transform.transformCloud(cloud.T).T

            # histogram equalization
            if not args.no_heq:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

            # draw cloud on image
            overlay_img = img.copy()
            overlay_img = draw_point_cloud(overlay_img, intrinsics, cloud, z_min=args.z_min, z_max=args.z_max,
                                           radius=args.radius)
            mixed_img = cv2.addWeighted(overlay_img, args.alpha, img, 1.0 - args.alpha, gamma=0.0)

            # visualize
            plt.imshow(mixed_img)
            plt.show()


if __name__ == '__main__':
    main()
