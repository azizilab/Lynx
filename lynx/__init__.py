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

# Make the underlying flat modules importable (they rely on bare imports
# such as ``import utils`` / ``import configs`` internally).
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_root, os.path.join(_root, "models"), os.path.join(_root, "util")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import (  # noqa: E402
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

__version__ = "0.1.0"
