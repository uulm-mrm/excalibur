from typing import Generator, Optional, Tuple

import motion3d as m3d
import numpy as np

from rclpy.time import Time
np.float = float  # fix for ros2_numpy AttributeError
import ros2_numpy  # noqa: E402

from .reader import Reader  # noqa: E402


def msg_to_numpy(msg, fields: Optional[Tuple[str, ...]] = None):
    # convert cloud to numpy
    cloud = ros2_numpy.numpify(msg)

    # select specific fields
    if fields is not None:
        cloud = np.stack([cloud[name].astype(float) for name in fields], axis=-1)

    return cloud


def iterate_pointcloud2_data(reader: Reader, topic: str, fields: Optional[Tuple[str, ...]] = None)\
        -> Generator[Tuple[Time, np.ndarray], None, None]:
    # iterate messages
    for _, _, msg in reader.messages(topic):
        # convert and yield
        cloud = msg_to_numpy(msg, fields)
        stamp = m3d.Time.FromSecNSec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        yield stamp, cloud
