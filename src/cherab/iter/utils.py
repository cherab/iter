"""Utility functions for the CHERAB-ITER module."""

from pathlib import Path

from platformdirs import user_cache_path

__all__ = ["get_cache_path", "IMAS_DB_PREFIX"]


IMAS_DB_PREFIX = Path("/work/imas/shared/imasdb/")
"""Path: The prefix for the IMAS database path."""


def get_cache_path(path: str) -> Path:
    """Check if a specific path (file or directory) exists in the cache directory.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    Path
        The full path to the file or directory in the cache directory.
    """
    _path = user_cache_path("cherab/iter") / Path(path)
    if _path.is_dir():
        _path.mkdir(parents=True, exist_ok=True)
    else:
        _path.parent.mkdir(parents=True, exist_ok=True)
    return _path
