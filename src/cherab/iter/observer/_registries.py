"""Defines the default IMAS queries for loading ITER observer data."""

import sys
from typing import TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


# ------------------------
# === Type definitions ===
# ------------------------
class IMASQuery(TypedDict):
    """Defines the parameters for an IMAS query to load observer data."""

    db: NotRequired[str]
    """IMAS database to query."""
    pulse: NotRequired[int]
    """IMAS pulse number."""
    run: NotRequired[int]
    """IMAS run number."""
    version: NotRequired[int]
    """IMAS version number."""
    path: NotRequired[str]
    """Path to the IMAS data. If provided, takes precedence over other parameters."""


# -----------------------
# === Default Queries ===
# -----------------------
OBSERVER_QUERIES: dict[str, IMASQuery] = {
    "bolometer": {
        "db": "ITER_MD",
        "pulse": 150401,
        "run": 3,
        "version": 3,
    },
}
