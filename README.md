# CHERAB-ITER

|         |                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![pre-commit.ci status][pre-commit-ci-badge]][pre-commit-ci] [![PyPI Publish][pypi-publish-badge]][pypi-publish]   |
| Docs    | [![Documentation Status][docs-badge]][docs]                                                                         |
| Package | [![PyPI - Version][pypi-badge]][pypi] [![Conda][conda-badge]][conda] [![PyPI - Python Version][python-badge]][pypi] |
| Meta    | [![License - EUPL-1.1][license-badge]][license] [![Pixi Badge][pixi-badge]][pixi-url]                               |

______________________________________________________________________

CHERAB-ITER; the CHERAB subpacakge for the ITER tokamak.

## 🔧 Get started for Developers

### Pre-requisites

- [`pixi`](pixi-url), a tool for project and package management.

If you don't have `git` installed, you can install it through `pixi` global installation:

```bash
pixi global install git
```

### Download and Run tasks

You can clone the repository and enter the directory with:

```bash
git clone https://github.com/cherab/iter
cd iter
```

Then, you can run tasks with `pixi` like:

```bash
pixi run <task>
```

For example, to run the tests:

```bash
pixi run test
```

Any other command can be seen with:

```bash
pixi task list
```

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

The documentation will be available at [Read the Docs][docs].

## 📄 License

This project is licensed under the terms of the [EUPL-1.1][license].

[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-iter
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-iter?logo=conda-forge&style=flat-square
[docs]: https://cherab-iter.readthedocs.io/en/latest/?badge=latest
[docs-badge]: https://readthedocs.org/projects/cherab-iter/badge/?version=latest&style=flat-square
[license]: https://opensource.org/licenses/EUPL-1.1
[license-badge]: https://img.shields.io/badge/license-EUPL_1.1%20-blue?style=flat-square
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh
[pre-commit-ci]: https://results.pre-commit.ci/latest/github/cherab/iter/main
[pre-commit-ci-badge]: https://results.pre-commit.ci/badge/github/cherab/iter/main.svg
[pypi]: https://pypi.org/project/cherab-iter/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-iter?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[pypi-publish]: https://github.com/cherab/iter/actions/workflows/deploy-pypi.yml
[pypi-publish-badge]: https://img.shields.io/github/actions/workflow/status/cherab/iter/deploy-pypi.yml?style=flat-square&label=PyPI%20Publish&logo=github
[python-badge]: https://img.shields.io/pypi/pyversions/cherab-iter?logo=Python&logoColor=gold&style=flat-square
