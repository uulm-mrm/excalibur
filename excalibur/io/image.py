from pathlib import Path
from typing import Generator, Tuple, Union

import numpy as np
import motion3d as m3d

from excalibur.io.camera import CameraIntrinsics
from excalibur.io.opencv import IMAGE_EXTENSIONS, PASSTHROUGH, read_image
from excalibur.io.stamped import StampedFileIterator


class StampedImageIterator(StampedFileIterator):
    def __init__(self, path: Union[Path, str], glob_pattern: str = '*'):
        # initialize iterator
        super().__init__(path, extensions=IMAGE_EXTENSIONS, glob_pattern=glob_pattern)

        # read intrinsics
        intrinsics_path = self._path / 'intrinsics.yaml'
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Intrinsics file '{intrinsics_path}' not found.")
        self._intrinsics = CameraIntrinsics.load(intrinsics_path)

    def iter(self, desired_encoding=PASSTHROUGH)\
            -> Generator[Tuple[m3d.Time, np.ndarray, CameraIntrinsics], None, None]:
        # iterate stamped image files
        for stamp, image_file in super().iter():
            # read image
            img = read_image(str(image_file), desired_encoding=desired_encoding)

            # yield data
            yield stamp, img, self._intrinsics
