"""Utility functions for the CHERAB-ITER module."""

from pathlib import Path

from platformdirs import user_cache_path

__all__ = ["get_cache_path"]


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
    return user_cache_path("cherab/iter") / Path(path)
