from collections import defaultdict
from pathlib import Path

from setuptools import setup

# Include demos in a separate directory in the distribution as data_files.
demo_parent_path = Path("share/cherab/demos/iter")
data_files = defaultdict(list)
demos_source = Path("demos")
for item in demos_source.rglob("*"):
    if item.is_file():
        install_dir = demo_parent_path / item.parent.relative_to(demos_source)
        data_files[str(install_dir)].append(str(item))
data_files = list(data_files.items())


setup(
    data_files=data_files,
)
