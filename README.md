# CHERAB-ITER

| | |
| ------- | ------- |
| CI/CD   | [![pre-commit.ci status][pre-commit-ci-badge]][pre-commit-ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish]|
| Docs    | [![Documentation Status][Docs-badge]][Docs] |
| Package | [![PyPI - Version][PyPI-badge]][PyPI] [![Conda][Conda-badge]][Conda] [![PyPI - Python Version][Python-badge]][PyPI] |
| Meta    | [![License - EUPL-1.1][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url] |

[pre-commit-ci-badge]: https://results.pre-commit.ci/badge/github/cherab/iter/main.svg
[pre-commit-ci]: https://results.pre-commit.ci/latest/github/cherab/iter/main
[PyPI-publish-badge]: https://img.shields.io/github/actions/workflow/status/cherab/iter/deploy-pypi.yml?style=flat-square&label=PyPI%20Publish&logo=github
[PyPI-publish]: https://github.com/cherab/iter/actions/workflows/deploy-pypi.yml
[Docs-badge]: https://readthedocs.org/projects/cherab-iter/badge/?version=latest&style=flat-square
[Docs]: https://cherab-iter.readthedocs.io/en/latest/?badge=latest
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-iter?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[PyPI]: https://pypi.org/project/cherab-iter/
[Conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-iter?logo=conda-forge&style=flat-square
[Conda]: https://prefix.dev/channels/conda-forge/packages/cherab-iter
[Python-badge]: https://img.shields.io/pypi/pyversions/cherab-iter?logo=Python&logoColor=gold&style=flat-square
<!-- [DOI-badge]: https://zenodo.org/badge/DOI/ -->
<!-- [DOI]: https://doi.org/ -->
[License-badge]: https://img.shields.io/badge/license-EUPL_1.1%20-blue?style=flat-square
[License]: https://opensource.org/licenses/EUPL-1.1
[pixi-badge]:https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

----

CHERAB-ITER; the CHERAB subpacakge for the ITER tokamak.

## 🔧 Quick Installation for Developers

You can quickly install with [`Pixi`][pixi-url] tool:
```bash
git clone https://github.com/cherab/iter
cd iter
pixi install -e dev
```
Then, you can run the tests with:
```bash
pixi run run-pytest
```
Any other command can be run with `pixi run <command>`.

## 🌐 Installation (future release)

You can install the package from PyPI:
```bash
pip install cherab-iter
```

Or from Conda:
```bash
mamba install -c conda-forge cherab-iter
```

## 📝 Documentation

The documentation will be available at [Read the Docs][Docs].

## 📄 License

This project is licensed under the terms of the [EUPL-1.1][License].
