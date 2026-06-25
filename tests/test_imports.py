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
    # Version is single-sourced from pyproject.toml; lynx.__version__ reads it back
    # from the installed package metadata, so the two must agree (a mismatch means
    # the package was not reinstalled after a bump, or the metadata name drifted).
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    m = re.search(r'(?m)^version = "([^"]+)"', pyproject)
    assert m, "no version field found in pyproject.toml"
    assert lynx.__version__ == m.group(1), (
        f"lynx.__version__={lynx.__version__!r} != pyproject {m.group(1)!r} — "
        "reinstall with `pip install -e .` after a version bump"
    )


def test_all_matches_submodules():
    assert set(lynx.__all__) == set(SUBMODULES)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_importable(name):
    mod = importlib.import_module(f"lynx.{name}")
    assert getattr(lynx, name) is mod


@pytest.mark.parametrize("submodule, attr", PUBLIC_SYMBOLS)
def test_public_symbol_resolves(submodule, attr):
    assert hasattr(getattr(lynx, submodule), attr)
