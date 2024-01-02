import cv2
import numpy as np


PASSTHROUGH = 'passthrough'


CVDEPTH_TO_NUMPY_DEPTH = {
    cv2.CV_8U: 'uint8',
    cv2.CV_8S: 'int8',
    cv2.CV_16U: 'uint16',
    cv2.CV_16S: 'int16',
    cv2.CV_32S: 'int32',
    cv2.CV_32F: 'float32',
    cv2.CV_64F: 'float64',
}


ENCODING_TO_CVTYPE2 = {
    'BGR8': cv2.CV_8UC3,
    'MONO8': cv2.CV_8UC1,
    'RGB8': cv2.CV_8UC3,
    'MONO16': cv2.CV_16UC1,
    'BGR16': cv2.CV_16UC3,
    'RGB16': cv2.CV_16UC3,
    'BGRA8': cv2.CV_8UC4,
    'RGBA8': cv2.CV_8UC4,
    'BGRA16': cv2.CV_16UC4,
    'RGBA16': cv2.CV_16UC4,
    'BAYER_RGGB8': cv2.CV_8UC1,
    'BAYER_BGGR8': cv2.CV_8UC1,
    'BAYER_GBRG8': cv2.CV_8UC1,
    'BAYER_GRBG8': cv2.CV_8UC1,
    'BAYER_RGGB16': cv2.CV_16UC1,
    'BAYER_BGGR16': cv2.CV_16UC1,
    'BAYER_GBRG16': cv2.CV_16UC1,
    'BAYER_GRBG16': cv2.CV_16UC1,
    'YUV422': cv2.CV_8UC2,
}

CONVERSION_ENCODING = {
    'MONO8': 'GRAY',
    'MONO16': 'GRAY',
    'BGR8': 'BGR',
    'BGR16': 'BGR',
    'RGB8': 'RGB',
    'RGB16': 'RGB',
    'BGRA8': 'BGRA',
    'BGRA16': 'BGRA',
    'RGBA8': 'RGBA',
    'RGBA16': 'RGBA',
    'YUV422': 'YUV422',
    'BAYER_RGGB8': 'BAYER_RGGB',
    'BAYER_RGGB16': 'BAYER_RGGB',
    'BAYER_BGGR8': 'BAYER_BGGR',
    'BAYER_BGGR16': 'BAYER_BGGR',
    'BAYER_GBRG8': 'BAYER_GBRG',
    'BAYER_GBRG16': 'BAYER_GBRG',
    'BAYER_GRBG8': 'BAYER_GRBG',
    'BAYER_GRBG16': 'BAYER_GRBG',
}

IMAGE_EXTENSIONS = ['bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff']


class CvHelperError(TypeError):
    pass


def cvtype2_to_dtype_with_channels(cvtype):
    depth = cvtype & 0b111  # first three bits represent the depth
    channels = (cvtype >> 3) + 1  # other bits represent the channels
    return CVDEPTH_TO_NUMPY_DEPTH[depth], channels


def encoding_to_cvtype2(encoding: str):
    encoding = encoding.upper()

    # Check for the most common encodings first
    if encoding in ENCODING_TO_CVTYPE2:
        return ENCODING_TO_CVTYPE2[encoding]

    # Check all the generic content encodings
    try:
        return cv2.__getattribute__(f'CV_{encoding}')
    except AttributeError:
        raise CvHelperError(f"Unrecognized image encoding [{encoding}]")


def encoding_to_dtype_with_channels(encoding: str):
    return cvtype2_to_dtype_with_channels(encoding_to_cvtype2(encoding))


def get_conversion_encoding(encoding: str):
    encoding = encoding.upper()

    if encoding in CONVERSION_ENCODING:
        return CONVERSION_ENCODING[encoding]

    raise CvHelperError(f"Unsupported encoding [{encoding}]")


def convert_color(src: np.ndarray, encoding_in: str, encoding_out: str) -> np.ndarray:
    # check passthrough
    if encoding_out == PASSTHROUGH:
        return src

    # conversion code
    conv_env_in = get_conversion_encoding(encoding_in)
    conv_env_out = get_conversion_encoding(encoding_out)

    if conv_env_in == conv_env_out:
        return src

    try:
        code = cv2.__getattribute__(f'COLOR_{conv_env_in}2{conv_env_out}')
    except AttributeError:
        raise CvHelperError(f"Unsupported conversion from [{encoding_in}] to [{encoding_out}]")

    # convert
    return cv2.cvtColor(src, code)


def read_image(filename: str, desired_encoding: str = PASSTHROUGH) -> np.ndarray:
    # passthrough
    if desired_encoding == PASSTHROUGH:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return img

    # get desired number of channels (color vs grayscale)
    _, n_channels = cvtype2_to_dtype_with_channels(ENCODING_TO_CVTYPE2[desired_encoding])

    if n_channels == 1:
        # grayscale
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # returns MONO8
        input_type = 'MONO'

    else:
        # color
        img = cv2.imread(filename, cv2.IMREAD_COLOR)  # returns BGR8
        input_type = 'BGR'

    # get input depth
    if img.dtype == np.uint8:
        input_depth = '8'
    elif img.dtype == np.uint16:
        input_depth = '16'
    else:
        CvHelperError(f"Unsupported image input type: {img.dtype}")

    # convert to desired encoding
    input_encoding = f'{input_type}{input_depth}'
    return convert_color(img, input_encoding, desired_encoding)
