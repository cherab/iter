"""Subpackage for handling JOREK datasets stored in IMAS.

This subpackage provides tools for working with JOREK datasets stored in IMAS and for creating
Raysect or Cherab objects from the data.
"""

import pickle

import numpy as np
from imas import DBEntry
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cherab.imas.ggd.unstruct_2d_extend_mesh import UnstructGrid2DExtended
from cherab.imas.ids.common.ggd import load_grid as _load_grid

from ..utility import BACKEND, IMAS_DB_PREFIX, get_cache_path

__all__ = ["IDS_QUERY", "load_grid"]

# Default ITER IMAS queries
IDS_QUERY = {
    "db": "ITER_DISRUPTIONS",
    "shot": 113113,
    "run": 2,
    "version": 4,
}


def load_grid(
    custom_imas_query: dict[str, str | int] | None = None,
    backend: BACKEND = "uda",
    ids: str = "radiation",
    num_toroidal: int | None = None,
    in_wall: bool = True,
    cache: bool = True,
    quiet: bool = False,
) -> UnstructGrid2DExtended:
    """Load the JOREK grid from the IMAS database.

    Parameters
    ----------
    custom_imas_query : dict[str, str | int] | None, optional
        Custom IMAS query parameters to override the defaults.
        Example:
            custom_imas_query = {
                "db": "ITER_DISRUPTIONS",
                "shot": 113113,
                "run": 2,
                "version": 4,
                "path": "/work/imas/shared/imasdb/ITER_DISRUPTIONS/4/113113/2",
            }
        The `path` key is optional. If provided, it takes precedence over other keys.
    backend : {"uda", "hdf5"}, optional
        The backend to use for loading the data. Default is "uda".
    ids : str, optional
        The IMAS IDS to load the grid from. Default is "radiation".
    num_toroidal : int | None, optional
        The number of toroidal segments to extend. Default is 64.
    in_wall : bool, optional
        If True, only grid cells inside the machine wall are loaded. Default is True.
    cache : bool, optional
        If True and the grid is already cached, load it from the cache.
        If True and the grid is not cached, save it to the cache after loading from IMAS.
        If False, always load from IMAS. Default is True.
    quiet : bool, optional
        If True, suppress console output. Default is False.

    Returns
    -------
    `~cherab.imas.ggd.unstruct_2d_extend_mesh.UnstructGrid2DExtended`
        The loaded unstructured 2D grid object.
    """
    if custom_imas_query is not None:
        query = IDS_QUERY | custom_imas_query
    else:
        query = IDS_QUERY

    db, shot, run, version = (
        query["db"],
        query["shot"],
        query["run"],
        query["version"],
    )
    if "path" in query:
        query_text = f"(path=[bold cyan]{query['path']}[/bold cyan])"
    else:
        query_text = (
            f"shot=[bold cyan]{shot}[/bold cyan], "
            f"run=[bold cyan]{run}[/bold cyan], "
            f"database=[bold cyan]{db}[/bold cyan], "
            f"version=[bold cyan]{version}[/bold cyan], "
            f"N_toroidal=[bold cyan]{num_toroidal}[/bold cyan]"
        )

    # Create progress bar
    progress = Progress(
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
        SpinnerColumn("simpleDots"),
        TextColumn("{task.fields[detail]}"),
        console=Console(quiet=quiet),
    )
    task_id = progress.add_task("Loading Grid", total=1, start=True, detail="")
    with progress:
        cache_path = get_cache_path(
            f"jorek/grid"
            f"_{shot}"
            f"_{run}"
            f"_{db}"
            f"_{version}"
            f"{f'_{num_toroidal}' if num_toroidal is not None else ''}"
            f"{'_inside' if in_wall else ''}.pickle"
        )

        # Load grid from cached file
        if cache and cache_path.exists():
            progress.update(task_id, detail=f"(from cache {query_text})", refresh=True)
            with open(cache_path, "rb") as file:
                grid = pickle.load(file)

            progress.update(
                task_id,
                advance=1,
                description="Grid loaded",
                detail=f"(from cache {query_text})",
                refresh=True,
            )

            return grid

        # Load grid from IMAS database
        else:
            progress.update(task_id, detail=f"(from IMAS {query_text})", refresh=True)

            # Use the path from query
            if (_path := query.get("path", None)) is not None:
                uri = f"imas:{backend}?path={_path};backend=hdf5"

            # Construct the URI from the query parameters
            else:
                path = IMAS_DB_PREFIX / f"{db}/{version}/{shot}/{run}"
                uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"

            # Load the grid_ggd from IMAS
            with DBEntry(uri=uri, mode="r") as entry:
                grid_ggd = entry.partial_get(ids, "grid_ggd(0)")
                grid = _load_grid(grid_ggd, num_toroidal=num_toroidal)

        if in_wall:
            from cherab.core.math import PolygonMask2D

            from ..machine.wall import load_wall_outline

            progress.update(
                task_id, description="Extracting grid cells inside the machine wall", refresh=True
            )

            # Create a mask to determine if 2D points are inside the machine wall
            wall_outline = load_wall_outline()
            wall_outline = np.vstack((wall_outline["First Wall"], wall_outline["Divertor"][::-1]))
            mask = PolygonMask2D(np.ascontiguousarray(wall_outline))

            # Determine if the cells' centers are inside the wall
            cell_centres = grid.cell_centre[: grid.num_faces, :]
            r_coords = np.hypot(cell_centres[:, 0], cell_centres[:, 1])
            z_coords = cell_centres[:, 2]

            indices_inside_wall = []
            for i in range(grid.num_faces):
                if mask(r_coords[i], z_coords[i]):
                    indices_inside_wall.append(i)

            # Extract only the cells that are inside the wall
            grid = grid.subset_faces(indices_inside_wall, name=f"{grid.name} (inside wall)")

        # Save the grid to cache
        if cache:
            progress.update(task_id, description=f"Caching grid as {cache_path}", refresh=True)
            with open(cache_path, "wb") as file:
                pickle.dump(grid, file)
            progress.console.print(f"Grid loaded from IMAS {query_text}")
            progress.update(
                task_id,
                description=f"Cached grid as [bold purple]{cache_path}",
                detail="",
                refresh=True,
                advance=1,
            )

        else:
            progress.update(
                task_id,
                description="Grid loaded",
                detail=f"from IMAS {query_text}",
                refresh=True,
                advance=1,
            )

    return grid
