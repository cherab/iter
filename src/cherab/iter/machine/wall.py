"""This module provides functions to load ITER PFC meshes from the IMAS database."""

from __future__ import annotations

import numpy as np
from imas import DBEntry
from raysect.core.math import translate
from raysect.core.scenegraph._nodebase import _NodeBase  # type: ignore
from raysect.optical.library import RoughTungsten
from raysect.optical.material import (
    AbsorbingSurface,  # type: ignore
    Material,  # type: ignore
    NullMaterial,  # type: ignore
)
from raysect.primitive import Cylinder, Mesh, Subtract, Union
from raysect.primitive.csg import CSGPrimitive
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn
from rich.table import Table

from cherab.imas.ids.common import get_ids_time_slice
from cherab.imas.ids.wall import load_wall_2d, load_wall_3d

from ..utility import BACKEND, IMAS_DB_PREFIX, get_cache_path

__all__ = [
    "load_pfc_mesh",
    "load_wall_outline",
    "load_wall_absorber",
    "load_outline_mesh",
    "show_registries",
]

# Default ITER IMAS queries
PFC_QUERIES = {
    "first_wall": {
        "db": "ITER_MD",
        "shot": 116100,
        "run": 1001,
        "version": 3,
        "skip": False,
    },
    "divertor": {
        "db": "ITER_MD",
        "shot": 116100,
        "run": 2001,
        "version": 3,
    },
    "first_wall_fine": {
        "db": "ITER_MD",
        "shot": 116100,
        "run": 3001,
        "version": 3,
        "skip": True,
    },
}

WALL_OUTLINE_QUERY = {
    "db": "ITER_MD",
    "shot": 116000,
    "run": 5,
    "version": 3,
}

ROUGHNESS_W = 0.29

# Default material map
MAP_MATERIALS = {
    "first_wall": (AbsorbingSurface, None),
    "divertor": (RoughTungsten, ROUGHNESS_W),
}


def show_registries() -> None:
    """Display the default ITER IMAS queries.

    Examples
    --------
    >>> show_registries()
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
    ┃ Name            ┃ Database ┃  Shot  ┃ Run  ┃ Version ┃
    ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
    │ first_wall      │ ITER_MD  │ 116100 │ 1001 │    3    │
    │ divertor        │ ITER_MD  │ 116100 │ 2001 │    3    │
    │ first_wall_fine │ ITER_MD  │ 116100 │ 3001 │    3    │
    │ wall_outline    │ ITER_MD  │ 116000 │  5   │    3    │
    └─────────────────┴──────────┴────────┴──────┴─────────┘
    """
    table = Table(title="ITER IMAS Queries", show_footer=False)
    table.add_column("Name", justify="left", style="cyan")
    table.add_column("Database", justify="left", style="magenta")
    table.add_column("Shot", justify="center", style="green")
    table.add_column("Run", justify="center", style="yellow")
    table.add_column("Version", justify="center", style="blue")

    queries = PFC_QUERIES | dict(wall_outline=WALL_OUTLINE_QUERY)
    for name, query in queries.items():
        table.add_row(
            name, query["db"], str(query["shot"]), str(query["run"]), str(query["version"])
        )

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
    backend: BACKEND = "uda",
) -> dict[str, Mesh]:
    """Load ITER PFC meshes from the IMAS database.

    Parameters
    ----------
    custom_imas_queries : dict[str, dict[str, int]], optional
        Custom IMAS queries. Default is `None`.
        You can provide a custom query, for example:
            custom_imas_queries = {
                "first_wall": {
                    "db": "ITER_MD",
                    "shot": 116100,
                    "run": 1001,
                    "version": 3,
                    "skip": False,
                    "path": "/work/imas/shared/imasdb/ITER_MD/3/116100/1001",
                },
            }
        The `path` key is optional and, if provided, takes precedence over other keys.
    custom_material : dict[str, tuple[Material, float | None]], optional
        Custom material mapping. Default is `None`.
        For example:
            custom_material = {
                "first_wall": (RoughTungsten, 0.29),
            }
        The last value is the material roughness. If `None`, the material will be
        `~raysect.optical.material.absorber.AbsorbingSurface`.
    reflection : bool, optional
        Whether to use reflective materials. Default is `False` (absorbing).
    is_fine_mesh : bool, optional
        Whether to load the fine mesh for the first wall. Default is `False`.
    parent : `~raysect.core.scenegraph._nodebase._NodeBase` | None, optional
        Parent node in the Raysect scene-graph. Default is `None`.
    quiet : bool, optional
        If `True`, suppresses output. Default is `False`.
    cache : bool, optional
        If `True`, caches the ``*.rsm`` mesh data. Default is `True`.
        If cached data exists, it will be loaded from the cache.
    backend : {"hdf5", "uda"}, optional
        IMAS backend to use. Default is `"uda"`.

    Returns
    -------
    dict[str, `~raysect.primitive.mesh.mesh.Mesh`]
        Dictionary of PFC meshes.

    Examples
    --------
    If mesh data is already cached in the cache directory, you can simply run:

    >>> meshes = load_pfc_mesh()

    To use a local IMAS database:

    .. code-block:: python

        custom_imas_queries = {
            "first_wall": {
                "path": "/path/to/database/",
            },
        }
        meshes = load_pfc_mesh(
            custom_imas_queries=custom_imas_queries,
            cache=False,
            backend="hdf5",
        )
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
        table.add_column("Path (URI)", justify="left", style="magenta")
        table.add_column("Material", justify="center", style="green")
        table.add_column("Roughness", justify="center", style="yellow")
        table.add_column("Loaded", justify="center")

        # Create Group to show progress bar and table
        progress_group = Group(table, progress)
    else:
        progress_group = Group(progress)

    # Load meshes
    meshes = {}
    with Live(progress_group, auto_refresh=True, console=Console(quiet=quiet)) as live:
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
                db, shot, run, version = (
                    query["db"],
                    query["shot"],
                    query["run"],
                    query["version"],
                )
                cache_path = get_cache_path(f"machine/{mesh_name}_{shot}_{run}_{db}_{version}.rsm")
                if cache and cache_path.exists():
                    progress.update(task_id, description=f"{progress_text} (from cache)")
                    live.refresh()
                    meshes[mesh_name] = Mesh.from_file(
                        cache_path, parent=parent, material=material, name=mesh_name
                    )
                    uri = str(cache_path)
                else:
                    progress.update(task_id, description=f"{progress_text} (from IMAS database)")
                    live.refresh()
                    if (_path := query.get("path", None)) is not None:
                        uri = f"imas:{backend}?path={_path};backend=hdf5"
                    else:
                        path = IMAS_DB_PREFIX / f"{db}/{version}/{shot}/{run}"
                        uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"

                    entry = DBEntry(uri=uri, mode="r")
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
                        uri,
                        material_cls.__name__,
                        str(roughness),
                        _status,
                    )
                progress.advance(task_id)

        progress.update(task_id, visible=False)
        live.refresh()

    return meshes


def _load_wall_mesh(entry: DBEntry, parent: _NodeBase | None) -> dict[str, Mesh]:
    """Load the ITER wall mesh from the IMAS database.

    Parameters
    ----------
    entry : `~imas.db_entry.DBEntry`
        The IMAS database entry.
    parent : `~raysect.core.scenegraph._nodebase._NodeBase` | None
        The parent node in the Raysect scene-graph.

    Returns
    -------
    dict[str, `~raysect.primitive.mesh.mesh.Mesh`]
        Dictionary of wall mesh components.
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
    backend: BACKEND = "uda",
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load the ITER wall outline from IMAS.

    Parameters
    ----------
    custom_wall_query : dict[str, dict[str, int]], optional
        Custom wall outline query. Default is `None`.
        You can provide a custom query, for example:
            custom_wall_query = {
                "db": "ITER_MD",
                "shot": 116000,
                "run": 5,
                "version": 3,
                "path": "/work/imas/shared/imasdb/ITER_MD/3/116000/5",
            }
        The `path` key is optional and, if provided, takes precedence over other keys.
    backend : {"hdf5", "uda"}, optional
        IMAS backend to use. Default is `"uda"`.
    cache : bool, optional
        If `True`, caches the wall outline data. Default is `True`.
        If cached data exists, it will be loaded from the cache.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary containing the wall outline data.

    Examples
    --------
    .. code-block:: python

        custom_wall_query = {
            "path": "/path/to/database/",
        }
        wall_outline = load_wall_outline(custom_wall_query=custom_wall_query)
    """
    # Update the default query with a custom one if provided
    if custom_wall_query is not None:
        query = WALL_OUTLINE_QUERY | custom_wall_query
    else:
        query = WALL_OUTLINE_QUERY

    # Load wall outline
    db, shot, run, version = query["db"], query["shot"], query["run"], query["version"]
    cache_path = get_cache_path(f"machine/wall_outline_{shot}_{run}_{db}_{version}.npy")
    if cache and cache_path.exists():
        wall_outline = np.load(cache_path, allow_pickle=True).item()
    else:
        if (_path := query.get("path", None)) is not None:
            uri = f"imas:{backend}?path={_path};backend=hdf5"
        else:
            path = IMAS_DB_PREFIX / f"{db}/{version}/{shot}/{run}"
            uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"
        with DBEntry(uri=uri, mode="r") as entry:
            description2d = entry.partial_get("wall", "description_2d(0)")
            wall_outline = load_wall_2d(description2d)
            if cache:
                np.save(cache_path, wall_outline)

    return wall_outline


def load_wall_absorber(parent: _NodeBase | None = None, **kwargs) -> CSGPrimitive:
    """Load the ITER wall outline and create a wall absorber.

    This function creates an absorbing wall around the ITER first wall and divertor to terminate
    rays that hit the wall.

    Parameters
    ----------
    parent : `~raysect.core.scenegraph._nodebase._NodeBase`, optional
        The parent node in the Raysect scene-graph.
    **kwargs
        Additional keyword arguments to pass to `.load_wall_outline`.

    Returns
    -------
    `~raysect.primitive.csg.CSGPrimitive`
        The wall absorber.

    Examples
    --------
    >>> load_wall_absorber()
    <raysect.primitive.csg.Union at 0x107b8fe80>
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

    # Create the absorbing wall and return it
    return Union(
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


def load_outline_mesh(
    num_toroidal, parent=None, material=None, name="Wall Outline Surface", **kwargs
) -> Mesh:
    """Create a mesh from the wall outline.

    This function generates a mesh representing the ITER wall outline, connecting the first wall and
    divertor and extending it toroidally.

    Parameters
    ----------
    parent : `~raysect.core.scenegraph._nodebase._NodeBase`, optional
        Parent node in the Raysect scene-graph.
    material : `~raysect.optical.material.Material`, optional
        Material of the mesh. Default is `~raysect.optical.material.material.NullMaterial`.
    name : str, optional
        The name of the mesh. Default is `"Wall Outline Surface"`.

    Returns
    -------
    `~raysect.primitive.mesh.mesh.Mesh`
        Wall outline mesh.
    """
    if not isinstance(num_toroidal, int) or num_toroidal <= 0:
        raise ValueError("num_toroidal must be a positive integer.")

    if material is None:
        material = NullMaterial()

    # Load the wall outline
    outlines = load_wall_outline(**kwargs)
    outline = np.vstack((outlines["First Wall"], outlines["Divertor"][::-1]))

    num_polygon = outline.shape[0]
    vertices = np.empty((num_polygon * num_toroidal, 3))
    triangles = np.empty((num_polygon * 2 * num_toroidal, 3), dtype=int)

    # Create vertices for the mesh
    for i_phi in range(num_toroidal):
        phi = i_phi * 2 * np.pi / num_toroidal

        vertices[i_phi * num_polygon : (i_phi + 1) * num_polygon, 0] = outline[:, 0] * np.cos(phi)
        vertices[i_phi * num_polygon : (i_phi + 1) * num_polygon, 1] = outline[:, 0] * np.sin(phi)
        vertices[i_phi * num_polygon : (i_phi + 1) * num_polygon, 2] = outline[:, 1]

    # Create indices for the triangles
    indices = np.arange(num_polygon * num_toroidal, dtype=int).reshape((num_toroidal, num_polygon))
    indices = np.pad(indices, ((0, 1), (0, 1)), mode="wrap")

    i_tri = 0
    for i, j in np.ndindex(num_toroidal, num_polygon):
        triangles[i_tri, 0] = indices[i, j]
        triangles[i_tri, 1] = indices[i + 1, j]
        triangles[i_tri, 2] = indices[i + 1, j + 1]

        i_tri += 1

        triangles[i_tri, 0] = indices[i, j]
        triangles[i_tri, 1] = indices[i + 1, j + 1]
        triangles[i_tri, 2] = indices[i, j + 1]

        i_tri += 1

    return Mesh(
        vertices=vertices,
        triangles=triangles,
        closed=True,
        parent=parent,
        material=material,
        name=name,
    )
