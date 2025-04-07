"""Calculate RTM for bolometer cameras.

This script calculates the Ray Transfer Matrix (RTM) for bolometer cameras.
The RTM is calculated for each foil of the bolometer camera.
The RTM is saved as a `*.nc` file in the `rtm` directory.
"""

# %%
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from raysect.optical import World, translate
from raysect.primitive import Cylinder
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cherab.iter.jorek import load_grid
from cherab.iter.machine import load_pfc_mesh, load_wall_absorber
from cherab.iter.observer.bolometry import load_bolometers
from cherab.iter.raytransfer import Discrete3DMeshRayTransferEmitter
from cherab.tools.raytransfer import RayTransferPipeline0D

BASE = Path(__file__).parent

# Get arguments
# -------------
parser = argparse.ArgumentParser(description="Compute RTM for bolometer cameras")
parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
args = parser.parse_args()
quiet = args.quiet

# Set console
console = Console(quiet=quiet)

# %%
# Create scene-graph
# ------------------
# Here we add objects (machine, emitters) to the scene root (World).

# Scene world
world = World()

# Machine mesh
meshes = load_pfc_mesh(parent=world, reflection=False, quiet=quiet, backend="hdf5")

# Wall Absorber
load_wall_absorber(parent=world)

# Ray Transfer Emitter
grid = load_grid(num_toroidal=None, quiet=quiet)
grid_name = grid.name
progress_rtm = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
    SpinnerColumn("simpleDots"),
    console=console,
)
progress_rtm_task = progress_rtm.add_task("Create Ray Transfer Emitter", total=1)
with progress_rtm:
    bins = grid.num_cell
    # Check the cached interpolator
    interpolation_file = BASE / f"{grid_name}_interpolator.pkl"
    if interpolation_file.exists():
        progress_rtm.update(
            progress_rtm_task, description="Loading Grid Interpolator from cache", refresh=True
        )
        with open(interpolation_file, "rb") as file:
            index_function = pickle.load(file)
    else:
        progress_rtm.update(
            progress_rtm_task, description="Creating Grid Interpolator", refresh=True
        )
        with open(interpolation_file, "wb") as file:
            index_function = grid.interpolator(np.arange(bins, dtype=np.double), fill_value=-1)
            pickle.dump(index_function, file)
    material = Discrete3DMeshRayTransferEmitter(index_function, bins, integration_step=0.005)
    eps = 1.0e-6  # ray must never leave the grid when passing through the volume
    radius = grid._mesh_extent["rmax"] - eps
    height = grid._mesh_extent["zmax"] - grid._mesh_extent["zmin"] + 2 * eps
    cylinder = Cylinder(
        radius,
        height,
        material=material,
        parent=world,
        transform=translate(0, 0, -0.5 * height),
        name="Ray Transfer Emitter",
    )
    progress_rtm.update(
        progress_rtm_task, description="Ray Transfer Emitter created!", advance=1, refresh=True
    )

# Bolometers
bolos = load_bolometers(world, quiet=quiet)

# %%
# Create Live progress
# --------------------
# Group of progress bars;
# Some are always visible, others will disappear when progress is complete
bolo_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
    console=console,
)
bolo_current_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
    console=console,
)
foil_progress = Progress(
    TextColumn("[bold blue]Progress for foils {task.percentage:.0f}%"),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} foil done)"),
    console=console,
)
# overall progress bar
overall_progress = Progress(
    TimeElapsedColumn(),
    BarColumn(),
    TextColumn("{task.description}"),
    console=console,
)
# Set progress panel
progress_panel = Group(
    Panel(
        Group(bolo_progress, bolo_current_progress, foil_progress),
        title="RTM for Bolometers",
        title_align="left",
        expand=False,
    ),
    overall_progress,
)

# Set tasks
num_tasks = len(bolos)
overall_task_id = overall_progress.add_task("", total=num_tasks)

# %%
# Compute RTM
# -----------
# Here we compute RTM for each bolometer camera.
# Each RTM is stored as an `xarray.Dataset` in a separate group named after the bolometer name.
# Then, the dataset is saved as a `*.nc` file in the `rtm` directory.
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_path = BASE / "rtm" / f"{time_now}.nc"
save_path.parent.mkdir(exist_ok=True)

with Live(progress_panel):
    # Calculate RTM for resistive bolometers
    i_task = 0
    for bolo in bolos:
        # Set overall task
        overall_progress.update(
            overall_task_id,
            description=f"[bold #AAAAAA]({i_task} out of {num_tasks} bolometers done)",
        )
        # Initialize current task
        current_task_id = bolo_current_progress.add_task("", action="Preprocessing")

        # List for storing RTM of each foil
        rtm = []

        # Set task for each progress
        bolo_task_id = bolo_progress.add_task(f"{bolo.name}")
        foil_task_id = foil_progress.add_task("", total=len(bolo))
        bolo_current_progress.update(current_task_id, action="Rendering")
        for foil in bolo:
            rtp = RayTransferPipeline0D(kind="power")
            foil.pipelines = [rtp]
            foil.pixel_samples = 1e4
            foil.min_wavelength = 600
            foil.max_wavelength = 601
            foil.spectral_rays = 1
            foil.spectral_bins = bins
            foil.observe()

            # Store values
            rtm += [rtp.matrix]

            # update progress
            foil_progress.update(foil_task_id, advance=1, refresh=True)

        foil_progress.update(foil_task_id, visible=False)

        bolo_current_progress.update(current_task_id, action="Saving")
        # Set config info
        config = {
            "bolometer name": bolo.name,
            "bolometer Number of foils": len(bolo),
            "foil spectral bins": bolo[0].spectral_bins,
            "foil pixel samples": bolo[0].pixel_samples,
            "foil wavelength range": (bolo[0].min_wavelength, bolo[0].max_wavelength),
        }
        xr.Dataset(
            data_vars=dict(
                rtm=(
                    ["foil", "voxel"],
                    np.asarray_chkfinite(rtm) / (4.0 * np.pi),  # [m^3 sr] -> [m^3]
                    dict(units="m^3", long_name="ray transfer matrix"),
                ),
                area=(
                    ["foil"],
                    np.asarray([foil.collection_area for foil in bolo]),
                    dict(units="m^2", long_name="foil area"),
                ),
            ),
            attrs=config,
        ).to_netcdf(save_path, mode="a", group=f"{bolo.name}")

        # update progress
        bolo_current_progress.stop_task(current_task_id)
        bolo_current_progress.update(current_task_id, visible=False)

        bolo_progress.stop_task(bolo_task_id)
        bolo_progress.update(bolo_task_id, description=f"[bold green]{bolo.name} done!")

        overall_progress.update(overall_task_id, advance=1, refresh=True)

        i_task += 1

    # Finalize progress
    overall_progress.update(
        overall_task_id,
        description=f"[bold green]{num_tasks} bolometers done!",
    )

# %%
# Post process
# ------------
# Save config info for the whole
config_total = {
    "primitives material": world.primitives[1].material.__class__.__name__,
    "machine PFCs": str(meshes),
    "grid name": grid_name,
    "grid RTM bins": bins,
}
xr.Dataset(
    coords=dict(
        foil=(
            ["foil"],
            np.arange(1, len(bolo) + 1, dtype=int),
            dict(units="ch", long_name="foil channel"),
        ),
        voxel=(
            ["voxel"],
            np.arange(bins, dtype=int),
            dict(long_name="voxel index"),
        ),
    ),
    attrs=config_total,
).to_netcdf(save_path, mode="a")
console.print(f"[bold green]RTM data saved at[/bold green] {save_path}")
