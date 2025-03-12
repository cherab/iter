:orphan:

.. _quickstart:

===========
Quick Start
===========

Only 3 steps to get started with `cherab-iter` with `Pixi`_.
Please make sure you have `Pixi`_ installed on your system.
If not, please follow the `Pixi installation guide <https://pixi.sh/latest#installation>`_.

1. Download the repository::

    git clone https://github.com/cherab/iter.git

2. Move to the repository::

    cd iter

3. Run the ``lab`` command::

    pixi run run-jupyterlab

Then, you can see the JupyterLab is automatically launched and enjoy some notebooks in the
`docs/notebooks/` directory.
If you don't see browser window opened, please find the URL like
(http://127.0.0.1:8888/lab?token=...) in the terminal output and open it in your browser.

.. figure:: ../_static/images/quickstart_jupyterlab.webp
   :align: center
   :alt: JupyterLab Window

   JupyterLab launched by `pixi run run-jupyterlab` command. You can execute notebooks in
   :ref:`Example Gallery <examples>`.


If you want to use the `cherab-iter` as a library, please follow the
:ref:`installation guide <installation>`.
