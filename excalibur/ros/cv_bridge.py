import sys
from typing import Union

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, Image

from excalibur.io.opencv import encoding_to_dtype_with_channels, convert_color, PASSTHROUGH


def imgmsg_to_cv2(img_msg: Union[Image, CompressedImage], desired_encoding: str = PASSTHROUGH) -> np.ndarray:
    # check compressed
    if isinstance(img_msg, CompressedImage):
        return compressed_imgmsg_to_cv2(img_msg, desired_encoding=desired_encoding)

    dtype, n_channels = encoding_to_dtype_with_channels(img_msg.encoding)
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        im = np.ndarray(shape=(img_msg.height, img_msg.width),
                        dtype=dtype, buffer=img_msg.data)
    else:
        if isinstance(img_msg.data, str):
            im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_msg.data.encode())
        else:
            im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_msg.data)
    # If the byte order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()

    # adjust encoding
    return convert_color(im, img_msg.encoding, desired_encoding)


def compressed_imgmsg_to_cv2(cmprs_img_msg: CompressedImage, desired_encoding: str = PASSTHROUGH) -> np.ndarray:
    # decode image
    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                     dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    # get input encoding from format
    input_encoding = cmprs_img_msg.format.split(';')[0]

    # adjust encoding
    return convert_color(im, input_encoding, desired_encoding)
