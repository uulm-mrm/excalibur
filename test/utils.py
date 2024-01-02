from pathlib import Path
import subprocess
from typing import Tuple, Union


def run_script(path: Union[str, Path], *args) -> Tuple[int, str]:
    args = [str(a) for a in args]
    result = subprocess.run([str(path), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.decode('utf8')
