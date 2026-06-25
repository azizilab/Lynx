"""Minimal tests for ``lynx.io.load_xenium``.

``load_xenium`` is the most-used entry point for spatial data. We exercise the
happy path against a synthetic ``cell_feature_matrix.h5`` (written as an AnnData
so the loader falls back from ``read_10x_h5`` to ``read_h5ad``) plus the basic
path-validation guard. No real Xenium data is required.
"""

import os

import pytest

import lynx


def _write_feature_matrix(adata, dirpath):
    """Persist a synthetic AnnData under the Xenium feature-matrix filename."""
    path = os.path.join(dirpath, "cell_feature_matrix.h5")
    adata.write_h5ad(path)
    return path


def test_load_xenium_roundtrip(make_adata, tmp_path):
    adata = make_adata(n_obs=40, n_vars=6)
    _write_feature_matrix(adata, str(tmp_path))

    # Disable count/cell filtering so the tiny matrix survives intact; spatial
    # coords are already present, so the loader skips the cells.csv.gz step.
    out = lynx.io.load_xenium(
        str(tmp_path),
        raw_count=True,
        min_counts=0,
        min_cells=0,
        load_metadata=True,
        load_img=False,
    )

    assert out.shape == adata.shape
    assert "spatial" in out.obsm_keys()
    assert out.obsm["spatial"].shape == (adata.n_obs, 2)
    # load_spatial_metadata populates the squidpy-style uns['spatial'] block.
    assert "spatial" in out.uns_keys()


def test_load_xenium_missing_path_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(AssertionError):
        lynx.io.load_xenium(missing)
