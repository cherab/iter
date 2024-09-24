from collections import defaultdict
from pathlib import Path
from setuptools import setup, find_packages


# Include demos in a separate directory in the distribution as data_files.
demo_parent_path = Path("share/cherab/demos/iter")
data_files = defaultdict(list)
demos_source = Path("demos")
for item in demos_source.rglob("*"):
    if item.is_file():
        install_dir = demo_parent_path / item.parent.relative_to(demos_source)
        data_files[str(install_dir)].append(str(item))
data_files = list(data_files.items())


with open("README.md") as f:
    long_description = f.read()


setup(
    name="cherab-iter",
    version="0.0.1",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    description="Cherab spectroscopy framework, Iter submodule",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/cherab",
    project_urls=dict(
        Tracker="https://github.com/cherab/iter/issues",
        Documentation="https://cherab.github.io/documentation/",
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['cherab'],
    include_package_data=True,
)
