import logging
from typing import Optional


# logger
logger = logging.getLogger('excalibur')
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)s:%(funcName)s) %(message)s'))
logger.addHandler(_stream_handler)


# context manager for temporary log level
class LogLevelContext:
    def __init__(self, level: Optional[int]):
        self._level = level

    def __enter__(self):
        self._prev_level = logger.level
        if self._level is not None:
            logger.setLevel(self._level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logger.setLevel(self._prev_level)
