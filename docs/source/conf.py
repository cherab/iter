# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from datetime import datetime

from packaging.version import parse

from cherab.iter import __version__

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cherab-iter"
author = "Cherab Team"
copyright = f"{datetime.now().year}, {author}"
version_obj = parse(__version__)
release = version_obj.base_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_codeautolink",
    "sphinx_github_style",
    "numpydoc",
]

default_role = "obj"

# autodoc config
autodoc_member_order = "bysource"

# autosummary config
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# numpydoc config
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# todo config
todo_include_todos = True

# sphinx_copybutton config
copybutton_exclude = ".linenos, .gp"
copybutton_only_copy_prompt_lines = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# This is added to the end of RST files — a good place to put substitutions to
# be used globally.
rst_epilog = ""
with open("common_links.rst") as cl:
    rst_epilog += cl.read()

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "common_links.rst",
]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# html_logo = "_static/images/cherab_lhd_logo.svg"

# html_favicon = "_static/favicon/favicon.ico"

# Define the json_url for our version switcher.
json_url = "https://cherab-iter.readthedocs.io/en/latest/_static/switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if version_obj.is_prerelease:
        version_match = "dev"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/cherab/iter",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "pygments_light_style": "default",
    "pygments_dark_style": "native",
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Cherab-ITER"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# === Intersphinx configuration ===============================================
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "raysect": ("http://www.raysect.org", None),
    "cherab": ("https://www.cherab.info", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

intersphinx_timeout = 10

# === NB Sphinx configuration =================================================
# nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::
        This page was generated from `{{ docname }}`__.
    __ https://github.com/cherab/iter/blob/main/docs/{{ docname }}
"""
nbsphinx_thumbnails = {
    # "notebooks/path/to/.ipnb": "_static/images/path/to/image.png",
}

# === sphinx_github_style configuration =======================================
# get tag name which exists in GitHub
tag = "main" if version_obj.is_devrelease else f"v{version_obj.public}"

# set sphinx_github_style options
top_level = "cherab"
linkcode_blob = tag
linkcode_url = "https://github.com/cherab/iter"
linkcode_link_text = "Source"
