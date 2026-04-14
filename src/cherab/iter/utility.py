"""Utility functions for the CHERAB-ITER module."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path

__all__ = ["get_cache_path", "IMAS_DB_PREFIX"]


IMAS_DB_PREFIX = Path("/work/imas/shared/imasdb/")
"""Path: The prefix for the IMAS database path.

In the case of the ITER SDCC, this is typically set to `/work/imas/shared/imasdb/`.
"""

BACKEND = Literal["hdf5", "uda"]
"""Literal: The supported backends for the IMAS database."""


def get_cache_path(path: str, mkdir: bool = False) -> Path:
    """Get the full path to the file or directory in the cache directory.

    Parameters
    ----------
    path
        The path to the file or directory in the cache directory, relative to the cache directory.
    mkdir
        If `True`, create the directory if it does not exist.

    Returns
    -------
    Path
        The full path to the file or directory in the cache directory.
    """
    _path = user_cache_path("cherab/iter") / Path(path)
    if mkdir:
        if _path.is_dir():
            _path.mkdir(parents=True, exist_ok=True)
        else:
            _path.parent.mkdir(parents=True, exist_ok=True)
    return _path
