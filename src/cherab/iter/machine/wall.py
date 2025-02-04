"""This module provides functions to load the ITER PFC meshes from the IMAS database."""

from __future__ import annotations

import numpy as np
from imas import DBEntry
from raysect.core.math import translate
from raysect.core.scenegraph._nodebase import _NodeBase  # type: ignore
from raysect.optical.library import RoughTungsten
from raysect.optical.material import (
    AbsorbingSurface,  # type: ignore
    Material,  # type: ignore
)
from raysect.primitive import Cylinder, Mesh, Subtract, Union
from raysect.primitive.csg import CSGPrimitive
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn
from rich.table import Table

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.wall import load_wall_2d, load_wall_3d

from ..utils import get_cache_path

__all__ = ["load_pfc_mesh", "load_wall_outline"]

# Default ITER IMAS quaries
PREFIX = "/work/imas/shared/imasdb/"
PFC_QUERIES = {
    "first_wall": {
        "db": "ITER_MD",
        "shot": 116100,
        "pulse": 1001,
        "version": 3,
        "skip": False,
    },
    "divertor": {
        "db": "ITER_MD",
        "shot": 116100,
        "pulse": 2001,
        "version": 3,
    },
    "first_wall_fine": {
        "db": "ITER_MD",
        "shot": 116100,
        "pulse": 3001,
        "version": 3,
        "skip": True,
    },
}

WALL_OUTLINE_QUERY = {
    "db": "ITER_MD",
    "shot": 116000,
    "pulse": 5,
    "version": 3,
}

ROUGHNESS_W = 0.29

# Default material map
MAP_MATERIALS = {
    "first_wall": (AbsorbingSurface, None),
    "divertor": (RoughTungsten, ROUGHNESS_W),
}


def show_registries() -> None:
    """Show Default ITER IMAS quaries."""
    table = Table(title="ITER IMAS Queries", show_footer=False)
    table.add_column("Name", justify="left", style="cyan")
    table.add_column("Database", justify="left", style="magenta")
    table.add_column("Shot", justify="center", style="green")
    table.add_column("Pulse", justify="center", style="yellow")
    table.add_column("Version", justify="center", style="blue")

    queries = PFC_QUERIES | WALL_OUTLINE_QUERY
    for name, query in queries.items():
        table.add_row(name, query["db"], query["shot"], query["pulse"], query["version"])

    console = Console()
    console.print(table)


def load_pfc_mesh(
    custom_imas_queries: dict[str, dict[str, int]] | None = None,
    custom_material: dict[str, tuple[Material, float | None]] | None = None,
    reflection: bool = False,
    is_fine_mesh: bool = False,
    parent: _NodeBase | None = None,
    quiet: bool = False,
    cache: bool = True,
    backend: str = "uda",
) -> dict[str, Mesh]:
    """Load the ITER PFC meshes from IMAS database.

    Parameters
    ----------
    custom_imas_queries : dict[str, dict[str, int]], optional
        The custom IMAS queries, by default `None`.
        If you can path the custom query, like:
        ```python
        custom_imas_queries = {
            "first_wall": {
                "db": "ITER_MD",
                "shot": 116100,
                "pulse": 1001,
                "version": 3,
                "skip": False,
                "path": "/work/imas/shared/imasdb/ITER_MD/3/116100/1001",
            },
        }
        ```
        `path` key is optional, if provided it is prioritized over the other keys.
    custom_material : dict[str, tuple[Material, float | None]], optional
        The custom material map, by default `None`.
        If you can path the custom material map, like:
        ```python
        custom_material = {
            "first_wall": (RoughTungsten, 0.29),
        }
        ```
        where the last value is the roughness of the material. If `None`, the material will be
        `AbsorbingSurface`.
    reflection : bool, optional
        Whether to use reflection or absorption, by default `False`.
    is_fine_mesh : bool, optional
        Whether to load the fine mesh of the first wall, by default `False`.
    parent : `~raysect.core.scenegraph._nodebase._NodeBase` | None, optional
        The parent node in the Raysect scene-graph, by default `None`.
    quiet : bool, optional
        Whether to suppress the output, by default `False`.
    cache : bool, optional
        Whether to cache the mesh data, by default `True`.
        If already cached, the data will be loaded from the cache.
    backend : {"hdf5", "mdsplus", "uda", "memory"}, optional
        The IMAS backend to use, by default `"uda"`.

    Returns
    -------
    dict[str, `~raysect.primitive.mesh.Mesh`]
        The PFC meshes.
    """
    # Merge user-defined queries with default queries
    if custom_imas_queries is not None:
        queries = PFC_QUERIES | custom_imas_queries
    else:
        queries = PFC_QUERIES

    # Merge user-defined materials with default materials
    if custom_material is not None:
        materials = MAP_MATERIALS | custom_material
    else:
        materials = MAP_MATERIALS

    # Update the first wall query if the fine mesh is requested
    if is_fine_mesh:
        queries["first_wall"]["skip"] = True
        queries["first_wall_fine"]["skip"] = False
        materials.setdefault("first_wall_fine", materials["first_wall"])

    # Create progress bar and add task
    progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), transient=True)
    task_id = progress.add_task("", total=len(queries))

    if not quiet:
        # Create Table of the status of loading
        table = Table(title="Plasma Facing Components", show_footer=False)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Path", justify="left", style="magenta")
        table.add_column("Material", justify="center", style="green")
        table.add_column("Roughness", justify="center", style="yellow")
        table.add_column("Loaded", justify="center")

        # Create Group to show progress bar and table
        progress_group = Group(table, progress)
    else:
        progress_group = Group(progress)

    # Load meshes
    meshes = {}
    with Live(progress_group, auto_refresh=False, console=Console(quiet=quiet)) as live:
        for mesh_name, query in queries.items():
            # Skip if the mesh is not requested
            if query.get("skip", False):
                continue

            progress_text = f"Loading {mesh_name}"
            progress.update(task_id, description=progress_text)
            live.refresh()
            try:
                # ================================
                # Configure material
                # ================================
                material_cls, roughness = materials[mesh_name]
                if not reflection:
                    material_cls = AbsorbingSurface
                    roughness = None

                if roughness is not None:
                    material = material_cls(roughness=roughness)
                else:
                    material = material_cls()

                # ================================
                # Load PFC Meshes
                # ================================
                db, shot, pulse, version = (
                    query["db"],
                    query["shot"],
                    query["pulse"],
                    query["version"],
                )
                cache_path = get_cache_path(f"machine/{mesh_name}_{shot}_{pulse}_{db}.rsm")
                if cache and cache_path.exists():
                    progress.update(task_id, description=f"{progress_text} (from cache)")
                    live.refresh()
                    meshes[mesh_name] = Mesh.from_file(
                        cache_path, parent=parent, material=material, name=mesh_name
                    )
                    path = str(cache_path)
                else:
                    progress.update(task_id, description=f"{progress_text} (from IMAS database)")
                    live.refresh()
                    if (_path := query.get("path", None)) is not None:
                        path = f"imas:{backend}?path={_path};backend=hdf5"
                    else:
                        path = f"imas:{backend}?path={PREFIX}/{db}/{version}/{shot}/{pulse};backend=hdf5"
                    entry = DBEntry(uri=path, mode="r")
                    meshes[mesh_name] = _load_wall_mesh(entry, parent).values()

                    # Cache the mesh
                    if cache:
                        meshes[mesh_name].save(cache_path)

                # Save the status of loading
                _status = "✅"
            except Exception as e:
                _status = f"❌ ({e})"
            finally:
                if not quiet:
                    table.add_row(
                        mesh_name,
                        path,
                        material_cls.__name__,
                        str(roughness),
                        _status,
                    )
                progress.advance(task_id)

        progress.update(task_id, visible=False)
        live.refresh()

    return meshes


def _load_wall_mesh(entry: DBEntry, parent: _NodeBase | None) -> dict[str, Mesh]:
    """Load the ITER wall mesh from IMAS database.

    Parameters
    ----------
    entry : `~imas.db_entry.DBEntry`
        The IMAS database entry.
    parent : `~raysect.core.scenegraph._nodebase._NodeBase` | None
        The parent node in the Raysect scene-graph.

    Returns
    -------
    dict[str, `~raysect.primitive.mesh.Mesh`]
        The wall mesh components.
    """
    wall_ids = get_ids_time_slice(entry, "wall")
    wall_dict = load_wall_3d(wall_ids.description_ggd[0])

    components = {}

    for key, value in wall_dict.items():
        mesh = Mesh(value["vertices"], value["triangles"], closed=False)
        mesh.parent = parent
        mesh.name = key
        components[key] = mesh

    return components


def load_wall_outline(
    custom_wall_query: dict[str, dict[str, int]] | None = None,
    backend: str = "uda",
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load the ITER wall outline from IMAS.

    Parameters
    ----------
    custom_wall_query : dict[str, dict[str, int]], optional
        The custom wall outline queries, by default `None`.
        If you can path the custom query, like:
        ```python
        custom_wall_query = {
            "db": "ITER_MD",
            "shot": 116000,
            "pulse": 5,
            "version": 3,
            "path": "/work/imas/shared/imasdb/ITER_MD/3/116000/5",
        }
        ```
        `path` key is optional, if provided it is prioritized over the other keys.
    backend : {"hdf5", "mdsplus", "uda", "memory"}, optional
        The IMAS backend to use, by default `"uda"`.
    cache : bool, optional
        Whether to cache the wall outline data, by default `True`.
        If already cached, the data will be loaded from the cache.

    Returns
    -------
    dict[str, numpy.ndarray]
        The wall outline data.
    """
    # Update the default queriey with custom one
    if custom_wall_query is not None:
        query = WALL_OUTLINE_QUERY | custom_wall_query
    else:
        query = WALL_OUTLINE_QUERY

    # Load wall outline
    db, shot, pulse, version = query["db"], query["shot"], query["pulse"], query["version"]
    cache_path = get_cache_path(f"machine/wall_outline_{shot}_{pulse}_{db}_{version}.npy")
    if cache and cache_path.exists():
        wall_outline = np.load(cache_path, allow_pickle=True).item()
    else:
        if (_path := query.get("path", None)) is not None:
            path = f"imas:{backend}?path={_path};backend=hdf5"
        else:
            path = f"imas:{backend}?path={PREFIX}/{db}/{version}/{shot}/{pulse};backend=hdf5"
        with DBEntry(uri=f"{path}", mode="r") as entry:
            description2d = entry.partial_get("wall", "description_2d(0)")
            wall_outline = load_wall_2d(description2d)
            if cache:
                np.save(cache_path, wall_outline)

    return wall_outline


def load_wall_absorber(parent: _NodeBase, **kwargs) -> CSGPrimitive:
    """Load the ITER wall outline and create the wall absorber.

    This function creates an absorbing wall around the ITER first walls and divertor to terminate
    the rays that hit the wall.

    Parameters
    ----------
    parent : `~raysect.core.scenegraph._nodebase._NodeBase`
        The parent node in the Raysect scene-graph.
    **kwargs
        The keyword arguments to pass to `.load_wall_outline`.

    Returns
    -------
    `~raysect.primitive.csg.CSGPrimitive`
        The wall absorber.
    """
    # Load the wall outline
    outlines = load_wall_outline(**kwargs)
    first_wall = outlines["First Wall"]
    divertor = outlines["Divertor"]

    # Get the limits
    rmin, rmax = np.min(first_wall[:, 0]), np.max(first_wall[:, 0])
    zmin, zmax = np.min(divertor[:, 1]), np.max(first_wall[:, 1])

    # Add some extension
    rmin *= 0.85
    rmax *= 1.15
    zmin *= 1.10
    zmax *= 1.10
    thickness = 0.1
    r_out = rmax + thickness
    hight_in = zmax - zmin
    hight_out = hight_in + thickness * 2

    # Create absorbe wall
    Union(
        Cylinder(rmin, hight_out),
        Subtract(
            Cylinder(r_out, hight_out),
            Cylinder(rmax, hight_in, transform=translate(0, 0, thickness)),
        ),
        transform=translate(0, 0, zmin - thickness),
        parent=parent,
        material=AbsorbingSurface(),
        name="Absorbing Wall",
    )
