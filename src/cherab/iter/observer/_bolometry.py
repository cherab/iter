"""Provides functionality to load bolometer cameras from the IMAS bolometer IDS."""

from imas import DBEntry
from raysect.core.scenegraph._nodebase import _NodeBase
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cherab.imas.observer import load_bolometers as imas_load_bolometers
from cherab.tools.observers import BolometerCamera

from ..utility import BACKEND, IMAS_DB_PREFIX, get_cache_path
from ._registries import OBSERVER_QUERIES, IMASQuery

__all__ = ["load_bolometers"]


def load_bolometers(
    custom_query: IMASQuery | None = None,
    parent: _NodeBase | None = None,
    backend: BACKEND = "uda",
    cache: bool = True,
    quiet: bool = False,
) -> list[BolometerCamera]:
    """Load ITER bolometer cameras from IMAS database.

    This function loads the bolometer camera data from the IMAS database using the specified query
    parameters, and creates the list of `~cherab.tools.observers.bolometry.BolometerCamera`.

    Parameters
    ----------
    custom_query
        Custom IMAS query to load the bolometer data. If None, the default query from
        `OBSERVER_QUERIES` will be used.
        You can also specify a custom query, for example:
            custom_query = {
                "db": "ITER_MD",
                "pulse": 150401,
                "run": 4,
                "version": 3,
                "path": "/path/to/imas/data",
            }
        The `path` parameter takes precedence over other parameters if provided, and is used to
        construct the IMAS URI directly.
    parent
        Parent node in the Raysect scene-graph, typically a `~raysect.optical.scenegraph.World` object.
    backend
        IMAS backend to use, by default `"uda"`.
    cache
        If True, the bolometer IDS will be cached locally after loading, and subsequent calls with
        the same query parameters will load from the cache instead of querying the IMAS database,
        by default True.
        The cache directory is determined by `.get_cache_path` function
    quiet
        If True, suppress the progress bar and table output when loading the bolometer cameras,
        by default False.

    Returns
    -------
    list[`~cherab.tools.observers.bolometry.BolometerCamera`]
        The bolometer cameras.

    Examples
    --------
    >>> from raysect.optical import World
    >>>
    >>> world = World()
    >>> bolos = load_bolometers(
    ...     custom_query={"db": "ITER_MD", "pulse": 150401, "run": 4, "version": 4},
    ...     parent=world,
    ...     backend="uda",
    ...     cache=True,
    ...     quiet=True,
    ... )
    >>> bolos
    """
    # Update the default query with a custom one if provided
    if custom_query is not None:
        query = OBSERVER_QUERIES["bolometer"] | custom_query
    else:
        query = OBSERVER_QUERIES["bolometer"]

    db, pulse, run, version = query["db"], query["pulse"], query["run"], query["version"]
    cache_path = get_cache_path(f"{db}/{version}/{pulse}/{run}/bolometer.h5")

    progress_text = "Loading bolometer cameras"
    if cache and cache_path.exists():
        uri = f"imas:hdf5?path={cache_path.parent.as_posix()}"
        progress_text += f" from cache ({uri})"
    else:
        if (_path := query.get("path", None)) is not None:
            uri = f"imas:{backend}?path={_path};backend=hdf5"
        else:
            path = IMAS_DB_PREFIX / f"{db}/{version}/{pulse}/{run}"
            uri = f"imas:{backend}?path={path.as_posix()};backend=hdf5"

        progress_text += f" from IMAS database ({uri})"

    if not quiet:
        # Output table of the loaded cameras
        table = Table(title="ITER Bolometer Cameras", show_footer=False)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("#Ch", justify="right", style="green")

        # Set up progress bar
        progress = Progress(
            SpinnerColumn(finished_text="✅"),
            TextColumn("[progress.description]{task.description}"),
        )
        task_id = progress.add_task(progress_text, total=1)

    else:
        table = _DummyTable()
        progress = _DummyProgress()
        task_id = None

    # Load the bolometer cameras
    with progress:
        bolometers = imas_load_bolometers(uri, "r", parent=parent)
        progress.advance(task_id) if task_id is not None else None

    # Cache the bolometer data
    if cache and not cache_path.exists():
        if not quiet:
            progress = Progress(
                SpinnerColumn(finished_text="✅"),
                TextColumn("[progress.description]{task.description}"),
            )
            task_id = progress.add_task(f"Caching bolometer data into {cache_path.parent}", total=1)
        with progress:
            with DBEntry(uri, "r") as entry:
                ids = entry.get("bolometer", autoconvert=False)
            with DBEntry(f"imas:hdf5?path={cache_path.parent.as_posix()}", "w") as entry:
                entry.put(ids)

            progress.advance(task_id) if task_id is not None else None

    # Output the table of loaded cameras
    if not quiet:
        for bolometer in bolometers:
            table.add_row(bolometer.name, str(len(bolometer)))
        console = Console()
        console.print(table)

    return bolometers


class _DummyTable:
    """A dummy table that does nothing, used when `quiet=True`."""

    def add_row(self, *args, **kwargs) -> None:
        pass


class _DummyProgress:
    """A dummy progress context manager that does nothing, used when `quiet=True`."""

    def __enter__(self) -> "_DummyProgress":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def add_task(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def advance(self, *args, **kwargs) -> None:
        pass
