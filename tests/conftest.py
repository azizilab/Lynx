"""Shared fixtures for the LYNX test suite.

The tests are deliberately self-contained: they synthesise tiny AnnData
objects in memory rather than depending on the (large, git-ignored) Xenium /
DESI inputs under ``data/`` or the snapshots under ``results/``. This keeps the
suite fast and runnable in CI without a GPU or any real data.
"""

import numpy as np
import pytest


def _grid_coords(n_obs: int) -> np.ndarray:
    """Roughly-square unit grid so the radius / kNN graph is well connected."""
    side = int(np.ceil(np.sqrt(n_obs)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    return np.c_[gx.ravel(), gy.ravel()][:n_obs].astype(float)


@pytest.fixture
def make_adata():
    """Factory producing a minimal spatial ``AnnData`` for graph/IO tests.

    Returns a callable so each test can request its own sizes / cluster setup
    without leaking state between tests.
    """
    import anndata as ad

    def _make(n_obs=40, n_vars=6, with_clusters=True, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = [f"gene{i}" for i in range(n_vars)]
        adata.obs_names = [str(i) for i in range(n_obs)]
        adata.obsm["spatial"] = _grid_coords(n_obs)
        if with_clusters:
            adata.obs["cell_type"] = np.array(["A", "B"])[rng.integers(0, 2, n_obs)]
        return adata

    return _make
