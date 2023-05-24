from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List


class MessageLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4


@dataclass
class Message:
    text: str
    level: MessageLevel


class MessageHandler:
    def __init__(self):
        self.msgs: List[Message] = []

    def debug(self, text: str):
        self.append(text, MessageLevel.DEBUG)

    def info(self, text: str):
        self.append(text, MessageLevel.INFO)

    def warn(self, text: str):
        self.append(text, MessageLevel.WARNING)

    def error(self, text: str):
        self.append(text, MessageLevel.ERROR)

    def fatal(self, text: str):
        self.append(text, MessageLevel.FATAL)

    def append(self, text: str, level: MessageLevel):
        self.msgs.append(Message(text=text, level=level))

    def extend(self, other: MessageHandler):
        self.msgs.extend(other.msgs)

    def print(self, level: MessageLevel = MessageLevel.DEBUG):
        for msg in self.msgs:
            if int(msg.level) < int(level):
                continue
            print(f"[{msg.level.name}] {msg.text}")


# logger
logger = logging.getLogger('excalibur')
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)s:%(funcName)s) %(message)s'))
logger.addHandler(_stream_handler)
