"""Sphinx configuration for the LYNX documentation site."""

import os
import re
import sys

# -- Path setup --------------------------------------------------------------
# conf.py lives in docs/; the repo root is one level up. Expose the repo root
# plus the flat models/ and util/ dirs so autodoc can import both the `lynx`
# namespace and the underlying modules (which use bare imports internally).
_root = os.path.abspath("..")
for _p in (_root, os.path.join(_root, "models"), os.path.join(_root, "util")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- Project information -----------------------------------------------------
project = "LYNX"
copyright = "2026, Azizi Lab"
author = "Azizi Lab"
# Single source of truth for the version: read it from pyproject.toml. The docs
# build does not install LYNX, so importlib.metadata is unavailable here — parse
# the file directly (it is always present in the checkout).
_m = re.search(
    r'(?m)^version = "([^"]+)"',
    open(os.path.join(_root, "pyproject.toml"), encoding="utf-8").read(),
)
release = _m.group(1) if _m else "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock the heavy / native stack so ReadTheDocs can build the API reference
# without installing torch, CUDA, or the imaging libraries. Lightweight deps
# (numpy/pandas/scipy/matplotlib/networkx) are kept real and installed via
# requirements-docs.txt.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchrl",
    "torch_geometric",
    "torch_sparse",
    "torch_scatter",
    "torch_cluster",
    "torch_spline_conv",
    "pyro",
    "squidpy",
    "scanpy",
    "anndata",
    "scFates",
    "cv2",
    "skimage",
    "tifffile",
    "spatialdata",
    "wandb",
    "ml_collections",
    "jenkspy",
    "pcha",
    "py_pcha",
    "sklearn",
    "statsmodels",
    "seaborn",
    "rpy2",
    # Pure-Python deps reached transitively through the lynx import chain
    # (IPython.display in util/utils.py, tqdm/lightning in models/base_model.py,
    # patsy in util/test_assoc.py, igraph/elpigraph in util/trajectory.py). They
    # are import-time only, so mocking keeps the RTD build torch- and dep-free.
    "IPython",
    "tqdm",
    "lightning",
    "patsy",
    "igraph",
    "elpigraph",
]

# -- Notebook handling -------------------------------------------------------
# The tutorial pipelines require large, gitignored data and a GPU, so the
# notebooks are never executed at build time; nbsphinx renders the outputs
# that were committed when the author ran them locally.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output (furo theme) ------------------------------------------------
html_theme = "furo"
html_title = "LYNX"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    # Theme-aware sidebar logo (filenames are relative to html_static_path).
    "light_logo": "lynx_logo.png",
    "dark_logo": "lynx_logo_dark.png",
}
