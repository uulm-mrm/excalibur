from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import motion3d as m3d


def path_to_time(path: Path) -> m3d.Time:
    # get stem without extension
    stem = path.stem

    try:
        # check for dot as seconds delimiter
        if '.' in stem:
            stem_sep = stem.split('.')
            if len(stem_sep) != 2:
                raise ValueError
            return m3d.Time.FromSecNSec(int(stem_sep[0]), int(stem_sep[1]))
        else:
            return m3d.Time.FromNSec(int(stem))
    except ValueError:
        raise ValueError(f"Invalid time in filename: '{path}'")


class StampedFileIterator:
    def __init__(self, path: Union[Path, str], extensions: Optional[List[str]] = None, glob_pattern: str = '*'):
        # store path
        self._path = Path(path)

        # make sure that extensions start with dot and are lowercase
        if extensions is not None:
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                          for ext in extensions]

        # select files with correct extensions
        stamped_files = [(path_to_time(p), p)
                         for p in self._path.glob(glob_pattern)
                         if extensions is None or p.suffix.lower() in extensions]

        # sort files by stamps
        self._file_tuples = sorted(stamped_files, key=lambda x: x[0])

        # get starting time and duration
        stamps = [stamp for stamp, _ in self._file_tuples]
        if len(stamped_files) > 0:
            self._starting_time = stamps[0].toNSec()
            self._duration = stamps[-1].toNSec() - stamps[0].toNSec()
        else:
            self._starting_time = None
            self._duration = 0

        # misc
        self._dt = None
        self._verbose = False
        self._latest_time = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def starting_time(self) -> Optional[int]:
        return self._starting_time

    @property
    def duration(self) -> int:
        return self._duration

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

    def iter(self) -> Generator[Tuple[m3d.Time, Path], None, None]:
        for stamp, path in self._file_tuples:
            # check dt
            stamp_nsec = stamp.toNSec()
            if self._latest_time is not None and self._dt is not None and stamp_nsec - self._latest_time < self._dt:
                continue

            # update time and print
            self._latest_time = stamp_nsec
            if self._verbose:
                time_since_start = self._latest_time - self._starting_time
                time_rel = time_since_start / self.duration if self.duration != 0.0 else 1.0
                print(f"{time_since_start * 1e-9:.1f} s / {self.duration * 1e-9:.1f} s  "
                      f"({time_rel * 1e2:.1f} %)")

            # yield result
            yield stamp, path
