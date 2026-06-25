"""LYNX — spatial trajectory inference with multi-omic integration.

This package provides a clean import namespace over the project's flat
``models/`` and ``util/`` modules. It re-exports the public API under
themed submodules so users can write::

    import lynx

    graph_data = lynx.dataset.HeteroDataset(...)
    model = lynx.model.HeteroAttnVGAE(...)
    lynx.trajectory.get_curve(adata)
    lynx.plot.disp_trajectory(adata)

The existing example scripts (``liver/``, ``breast/``, ``thymus/``) keep
their own ``sys.path``-based imports and are unaffected by this shim.
"""

import os
import sys
import warnings

# Make the underlying flat modules importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_root, os.path.join(_root, "models"), os.path.join(_root, "util")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# squidpy (imported transitively below) applies @njit decorators with a
# redundant ``nopython`` kwarg, which numba flags once at import time. 
with warnings.catch_warnings():  # noqa: E402
    warnings.filterwarnings(
        "ignore",
        message="nopython is set for njit and is ignored",
        category=RuntimeWarning,
    )
    from . import (
        config,
        dataset,
        io,
        model,
        plot,
        test_assoc,
        trajectory,
        utils,
    )

__all__ = [
    "model",
    "dataset",
    "config",
    "io",
    "plot",
    "utils",
    "trajectory",
    "test_assoc",
]

# The version lives in exactly one place — pyproject.toml. 
from importlib.metadata import PackageNotFoundError, version as _pkg_version  # noqa: E402

try:
    __version__ = _pkg_version("LYNX")
except PackageNotFoundError:  # imported from a source checkout without an install
    __version__ = "0.0.0"
