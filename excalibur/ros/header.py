from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time
from std_msgs.msg import Header


def stamp_to_nsec(stamp: TimeMsg) -> int:
    return Time.from_msg(stamp).nanoseconds


def get_header_from_time(stamp_ns: int, frame_id: str) -> Header:
    stamp = Time(sec=int(stamp_ns / 1e9),
                 nanosec=int(stamp_ns % 1e9))
    header = Header(stamp=stamp, frame_id=frame_id)
    return header
