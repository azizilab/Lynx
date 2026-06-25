"""Import smoke tests for the ``lynx`` namespace.

These guard the shim package wiring: a renamed/moved symbol in the underlying
``models/`` or ``util/`` modules, or a broken re-export, fails here before it
reaches a tutorial or a user.
"""

import importlib

import pytest

import lynx

# The themed submodules advertised by the package.
SUBMODULES = [
    "config",
    "dataset",
    "io",
    "model",
    "plot",
    "test_assoc",
    "trajectory",
    "utils",
]

# A handful of key public symbols the tutorials rely on (module, attribute).
PUBLIC_SYMBOLS = [
    ("model", "HeteroAttnVGAE"),
    ("dataset", "HeteroDataset"),
    ("dataset", "XeniumDataset"),
    ("io", "load_xenium"),
    ("io", "filter_cells"),
    ("trajectory", "get_curve"),
    ("trajectory", "get_tree"),
    ("trajectory", "compute_pseudotime"),
    ("utils", "get_zonations"),
    ("plot", "disp_trajectory"),
    ("test_assoc", "test_cci"),
]


def test_version():
    assert isinstance(lynx.__version__, str) and lynx.__version__


def test_all_matches_submodules():
    assert set(lynx.__all__) == set(SUBMODULES)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_importable(name):
    mod = importlib.import_module(f"lynx.{name}")
    assert getattr(lynx, name) is mod


@pytest.mark.parametrize("submodule, attr", PUBLIC_SYMBOLS)
def test_public_symbol_resolves(submodule, attr):
    assert hasattr(getattr(lynx, submodule), attr)
