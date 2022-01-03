# This file is part of DUNEdn by M. Rossi
import os
from pathlib import Path


def get_dunedn_path():
    root = Path(os.environ.get("DUNEDN_PATH"))
    if root is not None:
        return root
    else:
        error_msg = f"""
Please, make the environment variable DUNEDN_PATH point to the DUNEdn repository root directory"""
        raise RuntimeError(error_msg)
