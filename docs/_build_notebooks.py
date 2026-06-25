"""Generate the three tutorial notebooks for the LYNX docs.

This builder writes clean, unexecuted .ipynb files into docs/tutorials/.
Each notebook mirrors the corresponding run_Lynx.py + the *core* of
downstream.py (pathway-signature plots are intentionally omitted). The
training pipeline is shown as code; the plotting cells load a pre-saved
LYNX result snapshot so the author can execute once locally against
`results/` and commit the rendered figures.

Run:  env/bin/python docs/_build_notebooks.py
"""

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tutorials")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


# Shared setup cell, parameterised by the example sub-directory the notebook
# is "run" from. Tutorials keep the project's sys.path style so the bare
# modules resolve exactly as in the example scripts; `import lynx` exposes the
# same API under a clean namespace.
def setup_cell(extra_imports=""):
    return code(
        f"""
import os

import numpy as np
import scanpy as sc
import squidpy as sq

import torch
import torch.nn as nn
{extra_imports}
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({{"figure.dpi": 120, "savefig.dpi": 300}})

# LYNX exposes its entire API under a single namespace
# (install once with `pip install -e .`, see the installation guide).
import lynx
"""
    )


# ---------------------------------------------------------------------------
# Tutorial 1 — Liver (Xenium + DESI, principal curve)
# ---------------------------------------------------------------------------
def build_liver():
    cells = [
        md(
            """
# Liver — Xenium transcriptomics + DESI metabolomics

This tutorial runs LYNX on a single liver sample that pairs **Xenium spatial
transcriptomics** (the *reference* modality) with **DESI metabolomics** (the
*query* modality). LYNX learns a shared latent embedding, infers the
periportal → pericentral zonation gradient as a principal **curve**, discretises
it into zones, and summarises cell–cell interactions.

We show the full training pipeline as code, then load a pre-saved LYNX result
snapshot for the downstream analysis and plots.
"""
        ),
        setup_cell(),
        md("## 1. Load and pair the two modalities"),
        code(
            """
# Hyperparameters
n_subgraphs = 16
n_hidden = 32
n_latent = 6
n_epochs = 500
lr = 1e-2
r = 50
patience = 20

xenium_path = "../data/xenium/"
desi_path = "../data/desi/"
sample_id = "NIH_F5_proseg"

adata_xenium = lynx.io.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id + ".h5"))

# Keep only cells/pixels with a valid cross-modal mapping.
adata_xenium, adata_desi = lynx.io.filter_cells(adata_xenium, adata_desi, by="map")

cluster_key = "subtype"
cluster_labels = adata_xenium.obs[cluster_key].cat.categories  # individual cell types
"""
        ),
        md("## 2. Build the heterogeneous graph and configure the model"),
        code(
            """
graph_data = lynx.dataset.HeteroDataset(
    adatas_ref=adata_xenium,
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs,
    r=r,
    is_weighted=True,
    alpha=0.5,
    cluster_key=cluster_key,
)

train_configs = lynx.config.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience,
    device=torch.device("cuda"),
)

model_configs = lynx.config.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden,
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,   # enable cell-cell interaction inference
)
"""
        ),
        md("## 3. Train the model and reconstruct features"),
        code(
            """
model = lynx.model.HeteroAttnVGAE(model_configs, device=torch.device("cuda"))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_desi,
    graph_data=graph_data,
    device=torch.device("cpu"),
)

# Reconstructed gene expression
adata_xenium.layers["px"] = res["px"].copy()
"""
        ),
        md(
            """
### Reconstruction quality

Observed vs. reconstructed gene expression should fall along the diagonal.
"""
        ),
        code(
            """
lynx.plot.disp_kde_scatter(
    adata_xenium.X.A.flatten(),
    res.px.flatten(),
    subset_ratio=0.001,
    xlabel=r"Observation $log(1+x)$",
    ylabel=r"Reconstruction $log(1+x)$",
    title="Feature reconstruction (Human liver)",
)
"""
        ),
        md(
            """
## 4. Load the pre-saved result snapshot

Training above requires the full (large) Xenium/DESI inputs and a GPU. For the
downstream analysis we load a LYNX result snapshot — the same object the
training step produces, carrying the latent embedding `X_z`, the DESI mapping,
and the cell-type labels.
"""
        ),
        code(
            """
results_dir = "../results/liver/"
adata_xenium = sc.read_h5ad(os.path.join(results_dir, "LYNX_xenium_6_0512.h5ad"))
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id + ".h5"))
adata_xenium, adata_desi = lynx.io.filter_cells(adata_xenium, adata_desi, by="map")
adata_desi.obsm["X_z"] = np.load(
    os.path.join(results_dir, "LYNX_desi_6_0512.npy")
).astype(np.float32)
"""
        ),
        md(
            """
## 5. Trajectory inference (principal curve)

Fit a principal curve through the latent embedding and assign a continuous
pseudotime `t`. The root marker orients the gradient (here a periportal gene
for transcriptomics and a periportal metabolite for metabolomics).
"""
        ),
        code(
            """
curve = lynx.trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.5)
lynx.trajectory.compute_pseudotime(adata_xenium, curve, root_marker="DPT")

sq.pl.spatial_scatter(
    adata_xenium, color="t",
    cmap="RdBu_r", size=25, img=False,
    title="Inferred spatial gradient\\nLYNX (Xenium)",
)
lynx.plot.disp_trajectory(
    adata_xenium, cmap="RdBu_r",
    title="Inferred spatial gradient\\nLYNX embedding",
)
"""
        ),
        code(
            """
# The same curve fit on the DESI (metabolomics) modality.
curve = lynx.trajectory.get_curve(adata_desi, epg_lambda=0.01, trim_radius_ratio=0.5)
lynx.trajectory.compute_pseudotime(adata_desi, curve, root_marker="Taurine [M-H]-")

sq.pl.spatial_scatter(
    adata_desi, color="t",
    cmap="RdBu_r", size=1, img=False,
    title=r"Spatial gradient $(t)$" + "\\nLYNX (DESI)",
)
"""
        ),
        md(
            """
## 6. Zonation

Discretise the continuous gradient into zones and identify the genes /
metabolites that vary across them.
"""
        ),
        code(
            """
# Normalise raw counts if needed before zone-feature testing.
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

lynx.utils.get_zonation_features(
    adata_xenium, adata_desi,
    n_zones=4, sample_id=sample_id,
    abundance_test=True, show=False,
)

sq.pl.spatial_scatter(
    adata_xenium, color="zone",
    size=25, img=False, palette="Set3",
)
"""
        ),
        md(
            """
## 7. Cell–cell interactions

Summarise the inferred interactions across all cell types and test their
significance, then render the overall interaction map.
"""
        ),
        code(
            """
cci_df = lynx.plot.summarize_cell_interaction(
    adata_xenium,
    cluster_key=cluster_key,
    cluster_labels=cluster_labels,
    show_plot=False,
)
cci_df, pval_df = lynx.test_assoc.test_cci(
    adata_xenium, cci_df,
    cluster_key=cluster_key,
    cluster_labels=cluster_labels,
)

lynx.plot.disp_heatmap(
    pval_df,
    title="Summary of cell-cell interaction (Overall)\\n -log10(p-val)",
)
"""
        ),
        code(
            """
# Per-zone interaction networks
for cluster_id in sorted(adata_xenium.obs["zone"].unique()):
    adata_sub = adata_xenium[adata_xenium.obs["zone"] == cluster_id].copy()
    zone_cci_df = lynx.plot.summarize_cell_interaction(
        adata_sub, cluster_key=cluster_key, cluster_labels=cluster_labels,
        show_plot=False,
    )
    zone_cci_df, zone_pval_df = lynx.test_assoc.test_cci(
        adata_sub, zone_cci_df,
        cluster_key=cluster_key, cluster_labels=cluster_labels,
    )
    lynx.plot.netVisual_circle(
        zone_cci_df, vertex_size_max=20, figsize=(12, 12),
        title=f"Interaction strength (Zone {int(cluster_id)})",
    )
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# Tutorial 2 — Breast (Xenium + H&E, principal tree)
# ---------------------------------------------------------------------------
def build_breast():
    cells = [
        md(
            """
# Breast — Xenium transcriptomics + H&E histology

This tutorial runs LYNX on a triple-positive breast-cancer sample that pairs
**Xenium spatial transcriptomics** with **H&E histology image patches**. The
DCIS → invasive progression is best captured by a **branching** structure, so
here LYNX fits a principal **tree** (rather than a single curve) over the latent
embedding.

We show the training pipeline as code, then load a pre-saved result snapshot for
the downstream tree inference and plots. Pathway-signature analyses from the full
study (stromal subtyping, single-gene / signature dynamics) are intentionally
omitted to keep this minimal.
"""
        ),
        setup_cell(extra_imports="import scFates as scf\nimport pandas as pd\n"),
        md("## 1. Load and filter the paired modalities"),
        code(
            """
# Dataset specs
n_subgraphs = 16
k = 8
r = 50
# Model / training parameters
n_hidden, n_latent = 32, 6
n_epochs, lr, patience = 500, 1e-3, 20

data_path = "../data/breast/dcis_fov/"
adata_xenium = sc.read_h5ad(os.path.join(data_path, "cell_feature_matrix.h5"))
adata_he = sc.read_h5ad(os.path.join(data_path, "he_patches_norm.h5ad"))
cluster_key = "cell_type"

# Drop unlabeled / extremely rare / hybrid annotations, and unify labels.
rare_labels = adata_xenium.obs[cluster_key].value_counts()[
    adata_xenium.obs[cluster_key].value_counts() < 10
].index.to_list()
labeled_mask = np.logical_and(
    adata_xenium.obs[cluster_key] != "Unlabeled",
    ~adata_xenium.obs[cluster_key].isin(rare_labels),
)
hybrid_mask = adata_xenium.obs[cluster_key].str.contains("Hybrid", case=False)
labeled_mask = np.logical_and(labeled_mask, ~hybrid_mask)

adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype(str)
adata_xenium.obs.loc[adata_xenium.obs[cluster_key] == "DCIS_1"] = "DCIS"
adata_xenium.obs.loc[adata_xenium.obs[cluster_key] == "Prolif_Invasive_Tumor"] = "Invasive_Tumor"
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype("category")

adata_xenium = adata_xenium[labeled_mask].copy()
adata_xenium.obs.index = adata_xenium.obs.index.astype(int)
adata_he = adata_he[labeled_mask].copy()
patch_size = np.sqrt(adata_he.var.shape[0] // 3).astype(int)  # H&E patch side length
"""
        ),
        md("## 2. Build the graph, configure, and train"),
        code(
            """
graph_data = lynx.dataset.HeteroDataset(
    adatas_ref=adata_xenium,
    adatas_query=adata_he,
    n_subgraphs=n_subgraphs,
    k=k, r=r,
    is_weighted=True,
    cluster_key=cluster_key,
    alpha=0.5,
    query="HE", query_proj_key="spatial",
    ref="Xenium", ref_proj_key="spatial",
)

train_configs = lynx.config.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience,
    device=torch.device("cuda"),
)
model_configs = lynx.config.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, c_latent=n_latent,
    patch_size=patch_size,          # H&E image-patch branch
    act=nn.SiLU(),
    infer_cell_interaction=True,
)

model = lynx.model.HeteroAttnVGAE(model_configs, device=torch.device("cuda"))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_he,
    graph_data=graph_data,
    device=torch.device("cpu"),
)

lynx.plot.disp_kde_scatter(
    adata_xenium.X.A.flatten().copy(),
    res.px.flatten().copy(),
    xlabel=r"Observation $log(x+1)$",
    ylabel=r"Reconstruction $log(x+1)$",
    title="Feature reconstruction (human breast cancer)",
)
"""
        ),
        md(
            """
## 3. Load the pre-saved result snapshot

Load the LYNX output (latent embedding `X_z`, cell-type labels, and the
inferred cell-interaction tensor) to drive the downstream analysis.
"""
        ),
        code(
            """
data_path = "../results/breast/"
adata = sc.read_h5ad(os.path.join(data_path, "LYNX_xenium_cci2.h5ad"))

adata.obs[cluster_key] = adata.obs[cluster_key].astype("str")
adata.obs.loc[adata.obs[cluster_key] == "DCIS_1", cluster_key] = "DCIS"
adata.obs.loc[adata.obs[cluster_key] == "Prolif_Invasive_Tumor", cluster_key] = "Invasive_Tumor"
adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
"""
        ),
        md(
            """
## 4. Trajectory inference (principal tree)

Fit a principal tree over the latent embedding. `ppt_lambda` in `[1e3, 1e4]`
gives a smooth, simplified manifold.
"""
        ),
        code(
            """
principal_graph = lynx.trajectory.get_tree(
    adata,
    use_rep="X_z",
    n_nodes=int(0.01 * adata.n_obs),
    ppt_lambda=5e3,
    plot_graph=True,
)

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
scf.pl.graph(adata, basis="pca", ax=ax, title="Principal graph")
"""
        ),
        md(
            """
From the principal-tree visualisation we pick the root, the branching node, and
the two leaves, then compute pseudotime from the root.
"""
        ),
        code(
            """
root_node = 107
branch_node = 58
leave_nodes = [33, 41]
lynx.trajectory.compute_pseudotime(adata, principal_graph, source=root_node)
"""
        ),
        md("## 5. Spatial visualisations"),
        code(
            """
# Inferred spatial gradient (pseudotime) in tissue coordinates
fig, ax = plt.subplots(dpi=150)
sq.pl.spatial_scatter(
    adata, color="t", cmap="RdBu_r",
    size=20, img=False, edgecolor="none",
    ax=ax, return_ax=True, title="Inferred spatial gradient",
)
plt.show()

# Spatial cell-type map
fig, ax = plt.subplots(dpi=150)
sq.pl.spatial_scatter(
    adata, color=cluster_key,
    img=False, size=15, ax=ax, return_ax=True, title="",
)
plt.show()
"""
        ),
        md(
            """
## 6. Cell-type composition dynamics

Split the tree into the immune→DCIS and immune→invasive paths, then show how
cell-type composition shifts along each as a stacked dynamics plot.
"""
        ),
        code(
            """
# Assign tree segments (immune root / DCIS branch / invasive branch).
root_path = lynx.trajectory.sort_nodes(adata, root_node=root_node, term_node=branch_node)
dcis_path = lynx.trajectory.sort_nodes(adata, root_node=branch_node, term_node=leave_nodes[0])[1:]
invasive_path = lynx.trajectory.sort_nodes(adata, root_node=branch_node, term_node=leave_nodes[1])[1:]

principal_assignments = adata.obsm["X_R"].argmax(1)
segments = []
for assign in principal_assignments:
    if assign in root_path:
        segments.append("immune")
    elif assign in dcis_path:
        segments.append("dcis")
    else:
        segments.append("invasive")
adata.obs["zone"] = segments

adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

n_bins = 50
adata_dcis = adata_norm[adata_norm.obs["zone"].isin(["immune", "dcis"])].copy()
adata_dcis.obs_names = adata_dcis.obs_names.astype(str)
dcis_dynamic_df = lynx.utils.get_celltype_dynamics(
    adata_dcis, adata_dcis.obs[cluster_key], n_bins=n_bins
)

adata_invasive = adata_norm[adata_norm.obs["zone"].isin(["immune", "invasive"])].copy()
adata_invasive.obs_names = adata_invasive.obs_names.astype(str)
invasive_dynamic_df = lynx.utils.get_celltype_dynamics(
    adata_invasive, adata_invasive.obs[cluster_key], n_bins=n_bins
)
"""
        ),
        code(
            """
fig, ax = lynx.plot.disp_stacked_dynamics(
    dcis_dynamic_df,
    colors=adata.uns["cell_type_colors"],
    title="Cell-type dynamics (DCIS trajectory)",
)
fig, ax = lynx.plot.disp_stacked_dynamics(
    invasive_dynamic_df,
    colors=adata.uns["cell_type_colors"],
    title="Cell-type dynamics (Invasive trajectory)",
)
"""
        ),
        md("## 7. Overall cell–cell interactions"),
        code(
            """
cluster_labels = adata.obs[cluster_key].cat.categories
cci_df = lynx.plot.summarize_cell_interaction(
    adata, cluster_key=cluster_key,
    title="Interaction strength (Overall)", show_plot=False,
)
cci_df, pval_df = lynx.test_assoc.test_cci(adata, cci_df, cluster_labels, cluster_key=cluster_key)

fig, ax = lynx.plot.netVisual_circle(
    cci_df, figsize=(12, 12), min_threshold=0.0,
    colors=adata.uns["cell_type_colors"],
    title="Interaction strength (Overall)",
)
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# Tutorial 3 — Thymus (spatial RNA + protein, principal curve)
# ---------------------------------------------------------------------------
def build_thymus():
    cells = [
        md(
            """
# Thymus — spatial RNA + protein (CITE-seq)

This tutorial runs LYNX on a multi-cellular, high-throughput mouse-thymus
section that pairs **spatial RNA** with **protein abundance (CITE-seq)**. LYNX
reconstructs denoised gene expression, infers the cortex → medulla axis as a
principal **curve**, and discretises it into zones.

This is the most minimal of the three tutorials: it shows the training pipeline,
then loads a pre-saved snapshot for the trajectory and zonation plots.
"""
        ),
        setup_cell(),
        md("## 1. Load the paired RNA + protein modalities"),
        code(
            """
n_subgraphs, k = 16, 8
n_hidden, n_latent = 32, 6
n_epochs, lr, patience = 500, 1e-2, 20

data_path = "../data/thymus/"
sample_id = "Mouse_Thymus1"

adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, "adata_rna.h5"))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, "adata_protein.h5"))
adata_protein.var_names_make_unique()
cluster_key = "cell_type" if "cell_type" in adata_rna.obs.keys() else None
"""
        ),
        md(
            """
## 2. Build the grid graph, configure, and train

Both modalities are measured on a regular spatial grid, so we set
`is_ref_grid` / `is_query_grid`. Cell-interaction inference is left off here.
"""
        ),
        code(
            """
graph_data = lynx.dataset.HeteroDataset(
    adatas_ref=adata_rna,
    adatas_query=adata_protein,
    n_subgraphs=n_subgraphs,
    k=k, is_weighted=True,
    cluster_key=cluster_key,
    is_query_grid=True,
    is_ref_grid=True,
    query="protein", query_proj_key="spatial",
    ref="rna", ref_proj_key="spatial",
)

train_configs = lynx.config.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience,
    device=torch.device("cuda"),
)
model_configs = lynx.config.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=False,
)

model = lynx.model.HeteroAttnVGAE(model_configs, device=torch.device("cuda"))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_rna, adata_protein,
    graph_data=graph_data,
    n_subgraphs=1,
    device=torch.device("cpu"),
)
adata_rna.layers["px"] = res["px"].copy()

lynx.plot.disp_kde_scatter(
    adata_rna.X.flatten(),
    res.px.flatten(),
    xlabel=r"Observation log(x+1)",
    ylabel=r"Reconstruction log(x+1)",
    title="Spatial RNA feature reconstruction",
)
"""
        ),
        md(
            """
## 3. Load the pre-saved result snapshot

The snapshot carries the latent embedding `X_z` and spatial coordinates needed
for the trajectory and zonation below.
"""
        ),
        code(
            """
results_dir = "../results/thymus/"
adata_rna = sc.read_h5ad(os.path.join(results_dir, f"lynx_rna_6_{sample_id}.h5ad"))
"""
        ),
        md(
            """
## 4. Trajectory inference (principal curve)

Fit a principal curve through the latent embedding and orient the cortex →
medulla axis with a cortex marker (`Dcn`).
"""
        ),
        code(
            """
curve = lynx.trajectory.get_curve(adata_rna)
lynx.trajectory.compute_pseudotime(adata_rna, curve, root_marker="Dcn")

ax = sq.pl.spatial_scatter(
    adata_rna, color="t",
    cmap="RdBu_r", size=100, img=False, return_ax=True, title=None,
)
ax.set_title(r"Inferred spatial gradient $(t)$ - LYNX", fontsize=14)

lynx.plot.disp_trajectory(
    adata_rna, cmap="RdBu",
    title="Principal curve - LYNX",
)
"""
        ),
        md("## 5. Zonation"),
        code(
            """
if "milestones_colors" in adata_rna.uns_keys():
    adata_rna.uns.pop("milestones_colors")

lynx.utils.get_zonations(adata_rna, n_zones=4)
sq.pl.spatial_scatter(
    adata_rna, color="zone",
    size=100, img=False,
    title="Spatial clustering\\nLYNX (RNA)",
)
"""
        ),
    ]
    return cells


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    write_notebook("liver.ipynb", build_liver())
    write_notebook("breast.ipynb", build_breast())
    write_notebook("thymus.ipynb", build_thymus())


if __name__ == "__main__":
    main()
