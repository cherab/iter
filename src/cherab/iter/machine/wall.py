"""Provide functions to load ITER PFC meshes from the IMAS database."""

import numpy as np
from imas import DBEntry
from numpy.typing import NDArray
from raysect.core.math import translate
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.material import (
    AbsorbingSurface,
    Material,
    NullMaterial,
)
from raysect.primitive import Cylinder, Mesh, Subtract, Union
from raysect.primitive.csg import CSGPrimitive
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn
from rich.table import Table

from cherab.imas.wall import load_wall_mesh
from cherab.imas.wall import load_wall_outline as imas_load_wall_outline

from ..utility import BACKEND, IMAS_DB_PREFIX, get_cache_path
from ._registries import (
    MAP_MATERIALS,
    PFC_QUERIES,
    WALL_OUTLINE_QUERY,
    IMASQuery,
)

__all__ = [
    "load_pfc_mesh",
    "load_wall_outline",
    "load_wall_absorber",
    "load_outline_mesh",
    "show_registries",
]


def show_registries() -> None:
    """Display the default ITER IMAS queries.

    Examples
    --------
    >>> show_registries()
    ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
    ┃ Name            ┃ Database ┃ Pulse  ┃ Run  ┃ Version ┃
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
    table.add_column("Pulse", justify="center", style="green")
    table.add_column("Run", justify="center", style="yellow")
    table.add_column("Version", justify="center", style="blue")

    queries = PFC_QUERIES | dict(wall_outline=WALL_OUTLINE_QUERY)
    for name, query in queries.items():
        table.add_row(
            name,
            str(query.get("db", "")),
            str(query.get("pulse", "")),
            str(query.get("run", "")),
            str(query.get("version", "")),
        )

    console = Console()
    console.print(table)


def load_pfc_mesh(
    imas_queries: dict[str, IMASQuery] | None = None,
    material: dict[str, Material] | Material | None = None,
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
    imas_queries
        IMAS queries, by default is `None`.
        You can provide a custom query, for example:
            imas_queries = {
                "first_wall": {
                    "name": "FullTokamak.none.none",
                    "db": "ITER_MD",
                    "pulse": 116100,
                    "run": 1001,
                    "version": 3,
                    "skip": False,
                    "path": "/work/imas/shared/imasdb/ITER_MD/3/116100/1001",
                },
            }
        The `path` key is optional and, if provided, takes precedence over other keys.
    material
        Material mapping, by default is `None`.
        For example:
            material = {
                "first_wall": RoughTungsten(0.29),
            }
        If a single `Material` instance is provided, it will be used for all components,
        for example:
            material = NullMaterial()
        If `None`, the material will be determined by the default mapping defined in `.MAP_MATERIALS`.
    reflection
        Whether to use reflective materials, by default `False` (absorbing).
        If `False`, all materials will be set to `AbsorbingSurface()` regardless of the default
        mapping or custom material provided.
    is_fine_mesh
        Whether to load the fine mesh for the first wall, by default is `False`.
    parent
        Parent node in the Raysect scene-graph, by default is `None`.
    quiet
        If `True`, suppresses output, by default is `False`.
    cache
        If `True`, ``*.rsm`` mesh data will be stored, by default is `True`.
        The data will be stored in the cache directory defined by `.get_cache_path` with the same
        IMAS query structure, for example: `~/.cache/iter/ITER_MD/3/116100/1001/mesh.rsm`.
        If cached data exists, it will be loaded from the cache.
    backend
        IMAS backend to use, by default is `"uda"`.

    Returns
    -------
    dict[str, `~raysect.primitive.mesh.mesh.Mesh`]
        Dictionary of PFC meshes.

    Raises
    ------
    ValueError
        If `material` or `imas_queries` are not expected values.

    Examples
    --------
    If mesh data is already cached in the cache directory, you can simply run:

    >>> meshes = load_pfc_mesh()

    To use a local IMAS database:

    .. code-block:: python

        imas_queries = {
            "first_wall": {
                "path": "/path/to/database/",
            },
        }
        meshes = load_pfc_mesh(
            imas_queries=imas_queries,
            cache=False,
            backend="hdf5",
        )
    """
    # Merge user-defined queries with default queries
    if imas_queries is not None:
        queries = PFC_QUERIES | imas_queries
    else:
        queries = PFC_QUERIES

    # ------------------------
    # === Define materials ===
    # ------------------------
    materials: dict[str, Material] = {}
    if isinstance(material, Material):
        materials = {key: material for key in queries.keys()}
    else:
        if isinstance(material, dict):
            materials = MAP_MATERIALS | material
            for key, value in materials.items():
                if not isinstance(value, Material):
                    raise ValueError(
                        f"Invalid material for {key}: {value}. Must be a Material instance."
                    )
        elif material is None:
            materials = MAP_MATERIALS
        else:
            raise ValueError("`material` must be either a Material instance, a dict, or None.")

    if not reflection:
        materials = {key: AbsorbingSurface() for key in materials.keys()}

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

    # -----------------------
    # === Load PFC Meshes ===
    # -----------------------
    meshes: dict[str, Mesh] = {}
    uri = "N/A"
    with Live(progress_group, auto_refresh=True, console=Console(quiet=quiet)) as live:
        for mesh_name, query in queries.items():
            # Skip if the mesh is not requested
            if query.get("skip", False):
                continue

            progress_text = f"Loading {mesh_name}"
            progress.update(task_id, description=progress_text)
            live.refresh()
            try:
                db, pulse, run, version = (
                    query["db"],
                    query["pulse"],
                    query["run"],
                    query["version"],
                )
                cache_path = get_cache_path(f"{db}/{version}/{pulse}/{run}/mesh.rsm")
                if cache and cache_path.exists():
                    progress.update(task_id, description=f"{progress_text} (from cache)")
                    live.refresh()
                    meshes[mesh_name] = Mesh.from_file(
                        cache_path, parent=parent, material=materials[mesh_name], name=mesh_name
                    )
                    uri = str(cache_path)
                else:
                    progress.update(task_id, description=f"{progress_text} (from IMAS database)")
                    live.refresh()
                    if (_path := query.get("path", None)) is not None:
                        uri = f"imas:{backend}?path={_path};backend=hdf5"
                    else:
                        path = IMAS_DB_PREFIX / f"{db}/{version}/{pulse}/{run}"
                        uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"

                    meshes = load_wall_mesh(
                        uri,
                        "r",
                        parent=parent,
                        materials={query["name"]: materials[mesh_name]},
                    )

                    meshes = {mesh_name: meshes[query["name"]]}  # Keep only the requested mesh

                    # Cache the mesh
                    if cache:
                        meshes[mesh_name].save(cache_path)

                # Save the status of loading
                _status = "✅"
            except Exception as e:
                _status = f"❌ ({e})"
            finally:
                if not quiet:
                    roughness = getattr(materials[mesh_name], "roughness", None)

                    table.add_row(  # type: ignore
                        mesh_name,
                        uri,
                        materials[mesh_name].__class__.__name__,
                        str(roughness) if roughness is not None else "N/A",
                        _status,
                    )
                progress.advance(task_id)

        progress.update(task_id, visible=False)
        live.refresh()

    return meshes


def load_wall_outline(
    imas_query: IMASQuery | None = None,
    backend: BACKEND = "uda",
    cache: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Load the ITER wall outline from IMAS.

    Parameters
    ----------
    imas_query
        IMAS query, by default is `None`.
        You can provide a custom query, for example:
            imas_query = {
                "db": "ITER_MD",
                "pulse": 116000,
                "run": 5,
                "version": 3,
                "path": "/work/imas/shared/imasdb/ITER_MD/3/116000/5",
            }
        The `path` key is optional and, if provided, takes precedence over other keys.
    backend
        IMAS backend to use, by default is `"uda"`.
    cache
        If `True` and backend is `"uda"`, cache the wall ids data, by default is `True`.
        The data will be stored in the cache directory defined by `.get_cache_path` with the same
        IMAS query structure, for example: `~/.cache/iter/ITER_MD/3/116000/5/wall.h5`.
        If cached data exists, it will be loaded from the cache.

    Returns
    -------
    `dict[str, NDArray[np.float64]]`
        Dictionary containing the wall outline data.

    Examples
    --------
    .. code-block:: python

        imas_query = {
            "path": "/path/to/database/",
        }
        wall_outline = load_wall_outline(imas_query=imas_query)
    """
    # Update the default query with a custom one if provided
    if imas_query is not None:
        query = WALL_OUTLINE_QUERY | imas_query
    else:
        query = WALL_OUTLINE_QUERY

    # Load wall outline
    db, pulse, run, version = query["db"], query["pulse"], query["run"], query["version"]
    cache_path = get_cache_path(f"{db}/{version}/{pulse}/{run}/wall.h5")
    if cache and cache_path.exists():
        uri = f"imas:hdf5?path={cache_path.parent.as_posix()}"
    else:
        if (_path := query.get("path", None)) is not None:
            uri = f"imas:{backend}?path={_path};backend=hdf5"
        else:
            path = IMAS_DB_PREFIX / f"{db}/{version}/{pulse}/{run}"
            uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"

    # Load wall outline from IMAS
    wall_outline = imas_load_wall_outline(uri, "r")

    # Cache the wall outline
    if cache and backend == "uda" and not cache_path.exists():
        path = IMAS_DB_PREFIX / f"{db}/{version}/{pulse}/{run}"
        uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"
        with DBEntry(uri, "r") as entry:
            ids = entry.get("wall", autoconvert=False)
        with DBEntry(
            f"imas:hdf5?path={cache_path.parent.as_posix()}",
            "w",
            dd_version=str(ids.ids_properties.version_put.data_dictionary),
        ) as entry:
            entry.put(ids)

    return wall_outline


def load_wall_absorber(parent: _NodeBase | None = None, **kwargs) -> CSGPrimitive:
    """Load the ITER wall outline and create a wall absorber.

    This function creates an absorbing wall around the ITER first wall and divertor to terminate
    rays that hit the wall.

    Parameters
    ----------
    parent
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
    height_in = zmax - zmin
    height_out = height_in + thickness * 2

    # Create the absorbing wall and return it
    return Union(
        Cylinder(rmin, height_out),
        Subtract(
            Cylinder(r_out, height_out),
            Cylinder(rmax, height_in, transform=translate(0, 0, thickness)),
        ),
        transform=translate(0, 0, zmin - thickness),
        parent=parent,
        material=AbsorbingSurface(),
        name="Absorbing Wall",
    )


def load_outline_mesh(
    num_toroidal: int,
    parent: _NodeBase | None = None,
    material: None | Material = None,
    name: str = "Wall Outline Surface",
    **kwargs,
) -> Mesh:
    """Create a mesh from the wall outline.

    This function generates a mesh representing the ITER wall outline, connecting the first wall and
    divertor and extending it toroidally.

    Parameters
    ----------
    num_toroidal
        Number of toroidal segments to create.
    parent
        Parent node in the Raysect scene-graph.
    material
        Material of the mesh. Default is `~raysect.optical.material.material.NullMaterial`.
    name
        Name of the mesh. Default is `"Wall Outline Surface"`.

    Returns
    -------
    `~raysect.primitive.mesh.mesh.Mesh`
        Wall outline mesh.

    Raises
    ------
    ValueError
        If `num_toroidal` is not a positive integer.
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
