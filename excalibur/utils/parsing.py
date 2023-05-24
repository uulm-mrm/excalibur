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
