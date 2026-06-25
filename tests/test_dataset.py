"""Minimal construction tests for the graph datasets in ``models/dataset.py``.

Both classes build spatial (and, for ``HeteroDataset``, cross-modal) graphs from
AnnData. We construct each on tiny synthetic data with ``n_subgraphs=1`` (which
bypasses the metis ``ClusterData`` partitioning) and assert the resulting graph
objects carry the expected node/edge attributes. CPU-only, no GPU or real data.
"""

import torch
from torch_geometric.data import Data, HeteroData

import lynx


def test_xenium_dataset_builds(make_adata):
    adata = make_adata(n_obs=40, n_vars=6, with_clusters=True)
    ds = lynx.dataset.XeniumDataset(
        adata, k=8, r=50, n_subgraphs=1, cluster_key="cell_type", verbose=False,
    )

    assert len(ds) >= 1
    batch = ds.batches[0]
    assert isinstance(batch, Data)
    # Node features, edges, and per-node cluster codes are all populated.
    assert batch.x.shape == (adata.n_obs, adata.n_vars)
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.numel() > 0
    assert batch.cluster.shape == (adata.n_obs,)
    # Two cluster labels ("A"/"B") -> two codes.
    assert ds.num_clusters == 2


def test_hetero_dataset_builds(make_adata):
    adata_ref = make_adata(n_obs=40, n_vars=6, with_clusters=True, seed=0)
    adata_query = make_adata(n_obs=40, n_vars=4, with_clusters=False, seed=1)

    # Cross-modal projection coordinates (ref<->query) expected by HeteroDataset.
    adata_ref.obsm["desi_map"] = adata_ref.obsm["spatial"].copy()
    adata_query.obsm["xenium_map"] = adata_query.obsm["spatial"].copy()

    ds = lynx.dataset.HeteroDataset(
        adatas_ref=adata_ref,
        adatas_query=adata_query,
        k=8, r=50, r_bigraph=30, n_subgraphs=1,
        cluster_key="cell_type", verbose=False,
    )

    assert len(ds.hetero_batches) >= 1
    batch = ds.hetero_batches[0]
    assert isinstance(batch, HeteroData)
    # Default modality names are "Xenium" (ref) and "DESI" (query).
    assert "Xenium" in batch.node_types and "DESI" in batch.node_types
    assert isinstance(batch["Xenium"].x, torch.Tensor)
    # The cross-modal ref<->query edges exist.
    assert ("Xenium", "to", "DESI") in batch.edge_types
