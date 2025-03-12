:orphan:

.. _contribution:

============
Contribution
============

Contributions from the community are welcome.
Interested collaborators can make contact with Koyo Munechika (Core Developer) from the
`Source Repository`_.

.. include:: ../../../AUTHORS.md


For Developers
--------------
If you would like to develop this package, please fork the `GitHub Repository`_ at first, and follow
the :ref:`installation procedure<installation>`.

`Pixi`_ is required for several development tasks, such as building the documentation and running
the tests.
Please install it by following the `Pixi Installation Guide <https://pixi.sh/latest#installation>`_
in advance.

.. note::

    Before you start developing, please ensure that your code remains clean and consistent by installing pre-commit.
    Running the following command will automatically set up the hooks::

        pixi run pre-commit-install


.. tab-set::

    .. tab-item:: test

        To run the tests, you can do so with::

            pixi run test

    .. tab-item:: docs

        To build the documentation, you can do so with::

            pixi run doc-build

        The documentation will be built in the ``docs/build/html`` directory.

        If you want to clean the documentation, you can do so with::

            pixi run doc-clean

        If you want to host the documentation locally, you can do so with::

            pixi run doc-serve

    .. tab-item:: lint/format

        To lint the code, you can do so with::

            pixi run lint

        To format the code, you can do so with::

            pixi run format

        To run pre-commit hooks for all files, you can do so with::

            pixi run pre-commit-run

    .. tab-item:: ipython

        To run the IPython shell, you can do so with::

            pixi run ipython

        The IPython shell will be started with the package installed.


.. note::

    All registered commands can be shown by::

        pixi tasks list

.. note::

    If you have any questions or issues, please feel free to open an `Issue`_ in the `GitHub Repository`_.
