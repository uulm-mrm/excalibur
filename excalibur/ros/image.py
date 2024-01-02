from dataclasses import dataclass
from typing import Generator, Tuple

import motion3d as m3d
import numpy as np

from rclpy.time import Time
from sensor_msgs.msg import CameraInfo

from excalibur.io.camera import CameraIntrinsics
from excalibur.io.opencv import PASSTHROUGH

from .cv_bridge import imgmsg_to_cv2
from .reader import Reader


@dataclass
class CameraTopicsInfo:
    image_topic: str
    camera_info_topic: str
    is_rect: bool
    is_compressed: bool


def get_camera_topics_info(image_topic: str) -> CameraTopicsInfo:
    topic_parts = image_topic.split('/')

    # compression
    if topic_parts[-1] == 'compressed':
        topic_parts = topic_parts[:-1]
        is_compressed = True
    else:
        is_compressed = False

    # rect
    if topic_parts[-1] in 'image_rect':
        topic_parts = topic_parts[:-1]
        is_rect = True
    elif topic_parts[-1] in 'image':
        topic_parts = topic_parts[:-1]
        is_rect = False
    else:
        raise NotImplementedError(f"Unsupported image topic: {image_topic}")

    # camera info
    topic_parts.append('camera_info')
    camera_info_topic = '/'.join(topic_parts)

    return CameraTopicsInfo(
        image_topic=image_topic, camera_info_topic=camera_info_topic,
        is_rect=is_rect, is_compressed=is_compressed)


def camera_info_msg_to_intrinsics(camera_info_msg: CameraInfo, is_rect: bool) -> CameraIntrinsics:
    intrinsics = CameraIntrinsics(
        camera_matrix=camera_info_msg.p.reshape(3, 4)[:, :3],
        dist_coeffs=np.zeros(5) if is_rect else camera_info_msg.d,
    )
    return intrinsics


def iterate_image_data(reader: Reader, image_topic: str, desired_encoding: str = PASSTHROUGH,
                       queue_size: int = 20, slop: int = 1000)\
        -> Generator[Tuple[Time, np.ndarray, CameraIntrinsics], None, None]:
    # prepare topics
    camera_topics_info = get_camera_topics_info(image_topic)
    topics = [camera_topics_info.camera_info_topic, camera_topics_info.image_topic]

    # iterate messages
    for messages in reader.messages_sync(topics=topics, queue_size=queue_size, slop=slop):
        # convert intrinsics
        camera_info_msg = messages[camera_topics_info.camera_info_topic].msg
        intrinsics = camera_info_msg_to_intrinsics(camera_info_msg, camera_topics_info.is_rect)

        # convert image
        img_msg = messages[image_topic].msg
        img_stamp = m3d.Time.FromSecNSec(img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
        img = imgmsg_to_cv2(img_msg, desired_encoding=desired_encoding)

        # results
        yield img_stamp, img, intrinsics
