# noqa
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = "vkgdr"
copyright = "2022, National Research Foundation (SARAO)"

# Get the information from the installed package, no need to maintain it in
# multiple places.
try:
    dist = distribution(project)
except PackageNotFoundError:
    author = "Unknown author"
    release = "Unknown release"
else:
    author = dist.metadata["Author"]
    release = dist.metadata["Version"]
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "breathe",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = ["vkgdr._vkgdr", "pycuda"]  # Can't be built on readthedocs
autodoc_member_order = "bysource"

breathe_projects = {"vkgdr": "./doxygen/xml"}
breathe_default_project = "vkgdr"

intersphinx_mapping = {
    "pycuda": ("https://documen.tician.de/pycuda/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
