from pathlib import Path
from typing import Generator, Tuple, Union

import motion3d as m3d
import numpy as np

from excalibur.io.stamped import StampedFileIterator


CLOUD_EXTENSIONS = ['npy']


class StampedCloudIterator(StampedFileIterator):
    def __init__(self, path: Union[Path, str], glob_pattern: str = '*'):
        # initialize iterator
        super().__init__(path, extensions=CLOUD_EXTENSIONS, glob_pattern=glob_pattern)

    def iter(self) -> Generator[Tuple[m3d.Time, np.ndarray], None, None]:
        # iterate stamped image files
        for stamp, cloud_file in super().iter():
            # read cloud
            cloud = np.load(cloud_file)

            # yield data
            yield stamp, cloud
