:orphan:

.. _installation:

============
Installation
============

For Users
=========
`cherab-iter` can be installed by many package managers.
Explore the various methods below to install `cherab-iter` using your preferred package manager.

.. tab-set::

    .. tab-item:: pip

        ::

            pip install cherab-iter

    .. tab-item:: uv

        ::

            uv add cherab-iter


    .. tab-item:: conda

        ::

            conda install -c conda-forge cherab-iter

    .. tab-item:: pixi

        ::

            pixi add cherab-iter


For Developers
==============
If you want to install from source in order to contribute to develop `cherab-iter`,
`Pixi`_ is required for several development tasks, such as building the documentation and running the tests.
Please install it by following the `Pixi Installation Guide <https://pixi.sh/latest#installation>`_ in advance.

Afterwards, you can install `cherab-iter` by following three steps:

1. Clone the `cherab-iter` repository::

    git clone https://github.com/cherab/iter.git

2. Enter the repository directory::

    cd iter

3. Install the package::

    pixi install

`pixi` install required packages into the isolated environment, so you can develop `cherab-iter` without worrying about the dependencies.
To use cherab-iter in interactive mode, launch the Python interpreter by executing::

    pixi run python

Once the interpreter is running, you can import and use cherab-iter, for example::

    >>> from cherab.iter import __version__
    >>> print(__version__)

Additionally, useful commands for development are shown in the :ref:`contribution` section.
