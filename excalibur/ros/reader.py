from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import motion3d as m3d
import numpy as np

from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from excalibur.ros.header import stamp_to_nsec
from excalibur.utils.logging import logger


@dataclass
class MessagePair:
    stamp: int
    msg: Any


@dataclass
class TopicMetadata:
    name: str
    type: str


class ApproximateTimeSynchronizer:
    def __init__(self, topics: List[str], queue_size: int, slop: int = 0, use_bag_stamp: bool = False):
        self.queue_size = queue_size
        self.slop = slop
        self.use_bag_stamp = use_bag_stamp
        self.queues = {t: {} for t in topics}

    def add(self, topic: str, stamp: int, msg: Any) -> Optional[Dict[str, MessagePair]]:
        # handle single topic
        if len(self.queues) == 1:
            return {topic: MessagePair(stamp=stamp, msg=msg)}

        # check topic and select queue
        if topic not in self.queues:
            raise RuntimeError(f"Unexpected topic: '{topic}'")
        queue = self.queues[topic]

        # acquire stamp from header
        if not self.use_bag_stamp:
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = stamp_to_nsec(msg.header.stamp)
            else:
                raise RuntimeError("Message either has no header or no stamp.")

        # search for closest stamps in other queues
        best_stamps = {}
        for other_topic, other_queue in self.queues.items():
            # skip this topic
            if other_topic == topic:
                continue

            # compare to stamps in queue
            other_stamps = np.array(list(self.queues[other_topic].keys()))
            if len(other_stamps) == 0:
                best_stamps = None
                break
            stamp_diffs = np.abs(other_stamps - stamp)
            best_stamp_idx = np.argmin(stamp_diffs)

            # check slop
            if stamp_diffs[best_stamp_idx] <= self.slop:
                best_stamps[other_topic] = other_stamps[best_stamp_idx]
            else:
                best_stamps = None
                break

        # check if valid stamps are found for all topics
        if best_stamps is not None:
            # check slop for best stamps (stamp range)
            stamps = np.array(list(best_stamps.values()))
            min_stamp, max_stamp = np.min(stamps), np.max(stamps)
            if max_stamp - min_stamp <= self.slop:
                # valid collection: create best messages
                best_messages = {best_topic: MessagePair(stamp=best_stamp, msg=self.queues[best_topic][best_stamp])
                                 for best_topic, best_stamp in best_stamps.items()}

                # remove all returned messages from queues
                for best_topic, msg_pair in best_messages.items():
                    del self.queues[best_topic][msg_pair.stamp]

                # attach latest message and return
                best_messages[topic] = MessagePair(stamp=stamp, msg=msg)
                return best_messages

        # no valid collection found: add latest value to queue
        queue[stamp] = msg
        while len(queue) > self.queue_size:
            del queue[min(queue)]

        return None


class Reader:
    def __init__(self, path: Union[Path, str]):
        # store config
        self._path = Path(path)

        # initialize reader
        self._reader = rosbag2_py.SequentialReader()

        # misc
        self._dt = None
        self._verbose = False
        self._latest_stamp = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dt(self) -> int:
        return self._dt

    def set_dt(self, value: Optional[int]):
        self._dt = value

    @property
    def verbose(self) -> bool:
        return self._verbose

    def set_verbose(self, verbose: bool):
        self._verbose = verbose

    @property
    def starting_time(self) -> Time:
        return self._reader.get_metadata().starting_time

    @property
    def duration(self) -> Duration:
        return self._reader.get_metadata().duration

    def open(self):
        self._reader.open_uri(str(self._path))

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        return False

    def _messages(self, *topics: List[str]) -> Generator[Tuple[str, int, Any], None, None]:
        # check if given topics exist
        available_topics = [topic_metadata.name for topic_metadata in self._reader.get_all_topics_and_types()]
        for topic in topics:
            if topic not in available_topics:
                logger.warning(f"Topic '{topic}' not found in rosbag.")

        # configure topics
        self._reader.reset_filter()
        if len(topics) > 0:
            storage_filter = rosbag2_py.StorageFilter(topics=topics)
            self._reader.set_filter(storage_filter)

        # prepare message types for deserialization
        msg_types = {topic_metadata.name: get_message(topic_metadata.type)
                     for topic_metadata in self._reader.get_all_topics_and_types()
                     if topics is None or len(topics) == 0 or topic_metadata.name in topics}

        # iterate messages from beginning
        self._reader.seek(0)
        while self._reader.has_next():
            # read
            topic, data, t = self._reader.read_next()

            # deserialize and return
            msg = deserialize_message(data, msg_types[topic])
            yield topic, t, msg

    def _check_stamp_return(self, stamp: int) -> bool:
        # check passed time
        if self._latest_stamp is not None and self._dt is not None and stamp - self._latest_stamp < self._dt:
            return False

        # update stamp and print
        self._latest_stamp = stamp
        if self._verbose:
            time_since_start = self._latest_stamp - self.starting_time.nanoseconds
            time_rel = time_since_start / self.duration.nanoseconds if self.duration.nanoseconds != 0 else 1.0
            print(f"{time_since_start * 1e-9:.1f} s / {self.duration.nanoseconds * 1e-9:.1f} s  "
                  f"({time_rel * 1e2:.1f} %)")
        return True

    def messages(self, *topics: List[str]) -> Generator[Tuple[str, int, Any], None, None]:
        # iterate messages
        for topic, stamp, msg in self._messages(*topics):
            if self._check_stamp_return(stamp):
                yield topic, stamp, msg

    def messages_sync(self, topics: List[str], queue_size: int, slop: int, use_bag_stamp: bool = False)\
            -> Generator[Dict[str, MessagePair], None, None]:
        # initialize synchronizer
        synchronizer = ApproximateTimeSynchronizer(topics, queue_size, slop, use_bag_stamp)

        # iterate messages
        for topic, stamp, msg in self._messages(*topics):
            messages = synchronizer.add(topic, stamp, msg)
            if messages is not None and self._check_stamp_return(stamp):
                yield messages

    def topics(self):
        topics = [TopicMetadata(name=meta.name, type=meta.type)
                  for meta in self._reader.get_all_topics_and_types()]
        return topics


@dataclass
class StampData:
    bag: np.ndarray
    data: np.ndarray


class StampedReader(Reader):
    def __init__(self, path: Union[Path, str]):
        super().__init__(path)
        self._stamp_data = None
        self._msg_types = None

    @property
    def stamp_data(self):
        return self._stamp_data

    def _load_stamps(self):
        # reset filter
        self._reader.reset_filter()

        # initialize stamps
        self._stamp_data = {}

        # prepare message types for deserialization
        self._msg_types = {topic_metadata.name: get_message(topic_metadata.type)
                           for topic_metadata in self._reader.get_all_topics_and_types()}

        # iterate messages from beginning
        for topic, stamp, msg in self._messages():
            # header stamp
            if hasattr(msg, 'header'):
                data_stamp = m3d.Time.FromSecNSec(msg.header.stamp.sec, msg.header.stamp.nanosec).toNSec()
            else:
                data_stamp = stamp

            # store stamps
            if topic not in self._stamp_data:
                self._stamp_data[topic] = StampData(bag=[], data=[])
            self._stamp_data[topic].bag.append(stamp)
            self._stamp_data[topic].data.append(data_stamp)

        # convert stamp lists to arrays
        self._stamp_data = {topic: StampData(bag=np.array(data.bag), data=np.array(data.data))
                            for topic, data in self._stamp_data.items()}

    def open(self):
        super().open()
        self._load_stamps()

    def _find_closest_stamp(self, topic: str, data_stamp: int) -> Tuple[int, int, int]:
        stamp_idx = np.argmin(np.abs(self._stamp_data[topic].data - data_stamp))
        closest_bag_stamp = self._stamp_data[topic].bag[stamp_idx]
        closest_data_stamp = self._stamp_data[topic].data[stamp_idx]
        return stamp_idx, closest_bag_stamp, closest_data_stamp

    def message_at_stamp(self, topic: str, data_stamp: int):
        # topic filter
        self._reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

        # get bag stamp
        _, bag_stamp, _ = self._find_closest_stamp(topic, data_stamp)

        # seek to bag stamp
        self._reader.seek(bag_stamp)

        # get next message on topic
        if self._reader.has_next():
            # read
            topic, data, t = self._reader.read_next()

            # deserialize
            msg = deserialize_message(data, self._msg_types[topic])

            # data stamp
            if hasattr(msg, 'header'):
                data_stamp = m3d.Time.FromSecNSec(msg.header.stamp.sec, msg.header.stamp.nanosec).toNSec()
            else:
                data_stamp = t

            return data_stamp, msg
        else:
            return None, None


class IndexedReader(StampedReader):
    def __init__(self, path: Union[Path, str]):
        super().__init__(path)
        self._main_topic = None
        self._indexed_stamps = None

    @property
    def main_topic(self):
        return self._main_topic

    def set_main_topic(self, main_topic: str):
        # configure
        self._main_topic = main_topic

        # indexed stamps
        self._indexed_stamps = {main_topic: self._stamp_data[main_topic].data}

        for metadata in self.topics():
            # skip main topic
            if metadata.name == main_topic:
                continue

            # iterate main stamps
            self._indexed_stamps[metadata.name] = np.array([
                self._find_closest_stamp(metadata.name, data_stamp)[2]
                for data_stamp in self._stamp_data[main_topic].data
            ])

    def size(self) -> int:
        return len(self._indexed_stamps[self._main_topic])

    def get_index_for_stamp(self, data_stamp: int) -> int:
        return self._find_closest_stamp(self._main_topic, data_stamp)[0]

    def messages_at_index(self, topics: List[str], idx: int):
        msgs = {topic: self.message_at_stamp(topic, self._indexed_stamps[topic][idx])
                for topic in topics}
        return msgs
