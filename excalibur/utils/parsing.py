import argparse
from pathlib import Path


def dir_path(path: str) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid directory: '{path}'")


def file_path(path: str) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid file: '{path}'")


def pathlib_path(path: str) -> Path:
    return Path(path)


def valid_path(path: str) -> Path:
    path = Path(path)
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: '{path}'")


class ParseEnum(argparse.Action):
    def __init__(self, option_strings, enum_type, *args, **kwargs):
        self._enum_type = enum_type
        kwargs['choices'] = [f.name for f in list(enum_type)]
        if 'default' not in kwargs:
            kwargs['default'] = None
        super(ParseEnum, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            value = str(values[0])
        else:
            value = str(values)
        try:
            enum_value = self._enum_type[value]
            setattr(namespace, self.dest, enum_value)
        except KeyError:
            parser.error('Input {} is not a field of enum {}'.format(values, self._enum_type))


class ParsePybindEnum(argparse.Action):
    def __init__(self, option_strings, enum_type, *args, **kwargs):
        self._enum_type = enum_type
        kwargs['choices'] = [f for f in enum_type.__members__.keys()]
        if 'default' not in kwargs:
            kwargs['default'] = None
        super(ParsePybindEnum, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            value = str(values[0])
        else:
            value = str(values)
        try:
            enum_value = self._enum_type.__members__[value]
            setattr(namespace, self.dest, enum_value)
        except KeyError:
            parser.error('Input {} is not a field of enum {}'.format(values, self._enum_type))
