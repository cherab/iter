"""Defines the default IMAS queries and material mappings for loading ITER machine data."""

from __future__ import annotations

import sys
from typing import TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from raysect.optical.library import RoughTungsten
from raysect.optical.material import AbsorbingSurface, Material


# ------------------------
# === Type definitions ===
# ------------------------
class IMASQuery(TypedDict):
    """Defines the parameters for an IMAS query to load machine data."""

    name: NotRequired[str]
    """Name of the component, same as `cherab.imas.wall.load_wall_mesh` return keys.
    """
    db: NotRequired[str]
    """IMAS database to query."""
    pulse: NotRequired[int]
    """IMAS pulse number."""
    run: NotRequired[int]
    """IMAS run number."""
    version: NotRequired[int]
    """IMAS version number."""
    skip: NotRequired[bool]
    """Whether to skip this query."""
    path: NotRequired[str]
    """Path to the IMAS data. If provided, takes precedence over other parameters."""


# -----------------------
# === Default Queries ===
# -----------------------
PFC_QUERIES: dict[str, IMASQuery] = {
    "first_wall": {
        "name": "FullTokamak.none.none",
        "db": "ITER_MD",
        "pulse": 116100,
        "run": 1001,
        "version": 3,
        "skip": False,
    },
    "divertor": {
        "name": "Divertor.none.none",
        "db": "ITER_MD",
        "pulse": 116100,
        "run": 2001,
        "version": 3,
    },
    "first_wall_fine": {
        "name": "FullTokamak.none.none",
        "db": "ITER_MD",
        "pulse": 116100,
        "run": 3001,
        "version": 3,
        "skip": True,
    },
}

WALL_OUTLINE_QUERY: IMASQuery = {
    "db": "ITER_MD",
    "pulse": 116000,
    "run": 5,
    "version": 3,
}


# ------------------------
# === Material mapping ===
# ------------------------
ROUGHNESS_W = 0.29

MAP_MATERIALS: dict[str, Material] = {
    "first_wall": AbsorbingSurface(),
    "divertor": RoughTungsten(ROUGHNESS_W),
}
