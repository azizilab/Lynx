"""Minimal construction tests for the graph datasets in ``models/dataset.py``.

Both classes build spatial (and, for ``HeteroDataset``, cross-modal) graphs from
AnnData. We construct each on tiny synthetic data with ``n_subgraphs=1`` (which
bypasses the metis ``ClusterData`` partitioning) and assert the resulting graph
objects carry the expected node/edge attributes. CPU-only, no GPU or real data.

The model built on such a dataset (revised defaults and ``version='original'``)
and the interaction tests on its output (``util/test_assoc.py``) follow.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
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


def test_alpha_reaches_model_configs(make_adata):
    adata_ref = make_adata(n_obs=40, n_vars=6, with_clusters=True, seed=0)
    adata_query = make_adata(n_obs=40, n_vars=4, with_clusters=False, seed=1)
    adata_ref.obsm["desi_map"] = adata_ref.obsm["spatial"].copy()
    adata_query.obsm["xenium_map"] = adata_query.obsm["spatial"].copy()

    def build(**kwargs):
        return lynx.dataset.HeteroDataset(
            adatas_ref=adata_ref, adatas_query=adata_query,
            k=8, r=50, r_bigraph=30, n_subgraphs=1,
            cluster_key="cell_type", verbose=False, **kwargs,
        )

    # alpha set on the dataset is carried into the model config (including 0)
    assert lynx.config.set_model_configs(build(alpha=0.5)).alpha == 0.5
    assert lynx.config.set_model_configs(build(alpha=0.0)).alpha == 0.0
    # Default when alpha is set nowhere
    assert lynx.config.set_model_configs(build()).alpha == 1.0
    # A model-config kwarg wins
    assert lynx.config.set_model_configs(build(alpha=0.5), alpha=2.0).alpha == 2.0


def test_volume_prior(make_adata):
    """`depths` stacks the query modality into a registered volume (an absent depth is an empty slot) and switches the
    model to the 3D prior, which reads the stack at each pixel's own voxel."""
    refs, queries = [], []
    for s in range(3):
        ref = make_adata(n_obs=36, n_vars=6, with_clusters=True, seed=s)
        query = make_adata(n_obs=36, n_vars=4, with_clusters=False, seed=10 + s)
        ref.obsm["desi_map"] = ref.obsm["spatial"].copy()
        query.obsm["xenium_map"] = query.obsm["spatial"].copy()
        refs.append(ref), queries.append(query)

    ds = lynx.dataset.HeteroDataset(
        adatas_ref=refs, adatas_query=queries, k=8, r=3, r_bigraph=2, n_subgraphs=1,
        is_weighted=True, cluster_key="cell_type", depths=[0., 110., 330.], verbose=False,
    )
    # Depth slots every 110: the section at 330 sits in slot 3 and slot 2 is empty; the last channel is the mask
    assert ds.volume.shape[:2] == (5, 4) and ds.volume[-1].sum((1, 2)).tolist() == [36, 36, 0, 36]
    vox = ds.hetero_batches[1]["DESI"].vox
    assert torch.allclose(ds.volume[:-1, vox[:, 0], vox[:, 1], vox[:, 2]].T, ds.hetero_batches[1]["DESI"].x)

    torch.manual_seed(0)
    cpu = torch.device("cpu")
    model = lynx.model.HeteroAttnVGAE(lynx.config.set_model_configs(ds, act=nn.SiLU(), c_hidden=8, c_latent=3), device=cpu)
    assert type(model.prior).__name__ == "VolumePrior"
    svi, _, _ = model.setup(model, lynx.config.set_train_configs(n_epochs=1, device=cpu))
    assert np.isfinite(svi.step(ds[1]))

    model.evaluate(refs[1], queries[1], graph_data=ds, device=cpu)
    own = refs[1].obsm["X_z_prior"].copy()
    model.prior.volume[:, :-1, 0] += 1.  # the prior of a section draws on the neighbouring depth
    model.evaluate(refs[1], queries[1], graph_data=ds, device=cpu)
    assert np.isfinite(own).all() and not np.allclose(own, refs[1].obsm["X_z_prior"])


@pytest.fixture
def tissue(make_adata):
    """A small tissue: type A expresses genes 0-5, type B genes 6-11; a smooth niche field raises gene 0 in type A."""
    ref = make_adata(n_obs=144, n_vars=12, with_clusters=True, seed=0)
    rng = np.random.default_rng(0)
    a = (ref.obs["cell_type"] == "A").values[:, None]
    rate = np.where(a == (np.arange(12) < 6)[None], 20.0, 1.0)
    field = np.sin(ref.obsm["spatial"][:, 0] / 3.0)
    rate[:, 0] *= np.exp(0.8 * field)
    ref.X = rng.poisson(rate).astype(np.float32)
    ref.obs["field"] = field
    query = make_adata(n_obs=144, n_vars=4, with_clusters=False, seed=1)
    ref.obsm["desi_map"] = ref.obsm["spatial"].copy()
    query.obsm["xenium_map"] = query.obsm["spatial"].copy()
    ds = lynx.dataset.HeteroDataset(
        adatas_ref=ref, adatas_query=query, k=8, r=3, r_bigraph=2, n_subgraphs=1,
        is_weighted=True, cluster_key="cell_type", alpha=0.5, verbose=False,
    )
    return ref, query, ds


def _fit(ds, ref, query, **kwargs):
    """One training step and `model.evaluate`"""
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    model = lynx.model.HeteroAttnVGAE(lynx.config.set_model_configs(ds, act=nn.SiLU(), **kwargs), device=cpu)
    svi, _, _ = model.setup(model, lynx.config.set_train_configs(n_epochs=1, device=cpu))
    loss = svi.step(ds[0])
    model.evaluate(ref, query, graph_data=ds, device=cpu)
    return model, loss


def test_model_versions(tissue):
    """The revised model is the default; `version='original'` gives the published one"""
    ref, query, ds = tissue
    keys = ("omega_prior", "kappa_prior_scale", "kappa_guide", "niche_path", "hsic_weight", "alpha")
    revised, original = lynx.config.set_model_configs(ds), lynx.config.set_model_configs(ds, version="original")
    assert tuple(revised[k] for k in keys) == ("lognormal", "balanced", "posterior_solve", True, 0.0, 0.5)
    assert tuple(original[k] for k in keys) == ("exponential", "inverse", "amortized", False, 1e-3, 1.0)
    # One component switched back, the rest revised
    assert lynx.config.set_model_configs(ds, kappa_guide="amortized").niche_path


@pytest.mark.parametrize("version", ["revised", "original"])
def test_model_outputs(tissue, version):
    ref, query, ds = tissue
    model, loss = _fit(ds, ref, query, version=version)
    assert np.isfinite(loss)
    assert hasattr(model, "niche_proj") == (version == "revised")
    for k in ("X_z", "X_z_prior", "X_s_hat", "X_kappa", "X_msg", "omega", "abundance"):
        assert k in ref.obsm and np.isfinite(ref.obsm[k]).all()
    assert ref.uns["lynx"]["version"] == version
    assert ref.uns["lynx"]["jacobian"].shape == (2, 12, model.configs.c_latent)
    if version == "revised":
        assert model.kappa_solve_stats["uncertified"] == 0

    # `test_cci` runs the version's test: (strength of the significant pairs, -log10 q), sender x receiver
    ref.obs["cell_type"] = ref.obs["cell_type"].astype("category")
    ref.obs["t"] = ref.obsm["spatial"][:, 0] / ref.obsm["spatial"][:, 0].max()
    cci_df, qval_df = lynx.test_assoc.test_cci(ref, **(dict(B=10) if version == "revised" else {}))
    for df in (cci_df, qval_df):
        assert df.shape == (2, 2) and list(df.index) == list(df.columns) == ["A", "B"] and np.isfinite(df.values).all()
    assert ((cci_df.values > 0) == (qval_df.values > 0)).all() or version == "original"


def test_interaction_tests(tissue):
    ref, query, ds = tissue
    _fit(ds, ref, query)
    leak = lynx.test_assoc.fit_leak(ref.X, pd.Categorical(ref.obs["cell_type"]).codes, ref.obsm["spatial"], R=4)
    assert leak["L"].shape == ref.shape and (leak["L"] >= 0).all() and np.isfinite(leak["loglik"]).all()

    programs = {"A_own": list(ref.var_names[:6]), "B_own": list(ref.var_names[6:])}
    tests = lynx.test_assoc.get_test_interactions(ref, fields="field", programs=programs, B=20, leak=leak["L"])
    cxc, cxcxg, cxcxp = tests["cxc"], tests["cxcxg"], tests["cxcxp"]
    assert set(cxc.columns) >= {"sender", "receiver", "stat", "p", "q", "called", "exposed", "n_genes"} and len(cxc) == 2
    assert set(cxcxg.columns) >= {"sender", "receiver", "gene", "z", "p", "q", "significant"} and np.isfinite(cxcxg.z).all()
    assert set(cxcxp.columns) >= {"program", "z", "p", "q", "n_genes", "significant"} and set(cxcxp.program) == set(programs)
    # Genes are reported only where their pair is called
    assert not (cxcxg.significant & ~cxcxg.merge(cxc[["sender", "receiver", "called"]], on=["sender", "receiver"]).called).any()
    kxc, kxcxg, kxcxp, block = tests["kxc"], tests["kxcxg"], tests["kxcxp"], tests["kxcxg_block"]
    assert list(kxc.field.unique()) == ["field"] and len(kxc) == 2 and {"stat", "p", "q", "called"} <= set(kxc.columns)
    assert {"field", "receiver", "gene", "z", "p", "q", "significant"} <= set(kxcxg.columns)
    assert {"program", "z", "significant"} <= set(kxcxp.columns)
    assert {"receiver", "gene", "stat", "p", "q", "significant"} <= set(block.columns) and len(block) == len(kxcxg)
    assert block.p.between(0, 1).all()

    # The pair-level matrices from the tables already run
    cci_df, qval_df = lynx.test_assoc.test_cci(ref, tests=tests)
    called = cxc.set_index(["sender", "receiver"]).called
    assert all((cci_df.loc[a, b] > 0) == called[(a, b)] for a, b in called.index) and (np.diag(qval_df) == 0).all()

    # Group labels as the niche: one indicator per group
    ref.obs["zone"] = pd.Categorical(np.where(ref.obsm["spatial"][:, 1] > 5, "top", "bottom"))
    zones = lynx.test_assoc.get_test_interactions(ref, fields="zone", B=10, leak=leak["L"])["kxc"]
    assert set(zones.field) == {"top", "bottom"}

    # The display mask and the consensus over runs
    mask = lynx.test_assoc.cxcxg_mask(ref)
    assert mask.shape == (2, ref.n_vars) and mask.values.dtype == bool and list(mask.index.names) == ["sender", "receiver"]
    out = lynx.test_assoc.consensus([cxc, cxc, cxc], ["sender", "receiver"], need=2)
    assert (out.runs_called == 3 * cxc.called.values).all() and out.consensus.dtype == bool


def test_surrogate_fields():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, size=(400, 2))
    F = np.column_stack([np.sin(xy[:, 0] / 150.0), rng.normal(size=400)])
    S, recipes = lynx.test_assoc.surrogate_fields(F, xy, 5, np.random.default_rng(1))
    assert S.shape == (5, 400, 2) and len(recipes) == 2
    for b in range(5):  # Rank-matched: the same values, permuted
        assert np.allclose(np.sort(S[b, :, 0]), np.sort(F[:, 0])) and np.allclose(np.sort(S[b, :, 1]), np.sort(F[:, 1]))
    assert recipes[0][1] > recipes[1][1]  # The smooth column gets the smoother recipe (larger mixture weight)


def test_ligand_receptor_channels():
    from scipy.stats import norm

    genes = [f"g{i}" for i in range(40)]
    det = pd.DataFrame(0.5, index=["A", "B"], columns=genes)
    z = np.random.default_rng(0).normal(size=40)
    z[:10] += 3  # The ligand's targets respond
    cxcxg = pd.DataFrame(dict(sender="A", receiver="B", gene=genes, z=z, p=2 * norm.sf(np.abs(z)), significant=np.abs(z) > 3))
    prior = pd.DataFrame({"g30": np.r_[np.ones(10), np.zeros(30)] + 0.01 * np.arange(40)}, index=genes)
    lr_network = pd.DataFrame(dict(ligand=["g30"], receptor=["g31"]))
    lr = lynx.test_assoc.test_lr(cxcxg, prior, lr_network, det, B=500, n_null=500)
    row = lr[(lr.sender == "A") & (lr.receiver == "B")].iloc[0]
    assert row.significant and row.reported and row.receptors == "g31"

    import matplotlib
    matplotlib.use("Agg")
    mask = pd.DataFrame(False, index=pd.MultiIndex.from_tuples([("A", "B")], names=["sender", "receiver"]), columns=genes)
    mask.loc[("A", "B"), "g0"] = True
    fig, ax = lynx.plot.disp_interaction_strength(cxcxg, mask=mask)
    assert len(ax.get_yticklabels()) == int(cxcxg.significant.sum())
