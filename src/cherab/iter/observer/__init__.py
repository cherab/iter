"""Subpackage for ITER observers."""

from ._bolometry import load_bolometers
from ._registries import OBSERVER_QUERIES

__all__ = ["load_bolometers", "OBSERVER_QUERIES"]
