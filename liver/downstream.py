# %%
# ----------------------
#  Downstream analysis
# ----------------------

# %%
import os
import sys
import gc

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')
from util import IO, utils, plot, test_assoc, trajectory

# %%
from IPython.display import display

from matplotlib import rcParams
from matplotlib.axes import Axes
rcParams['font.family'] = 'Liberation Sans'
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 150})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

# %%
%load_ext autoreload
%autoreload 2

# %%
# Load data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
indir = '../results/'
outdir = '../results/downstream/'

# %%
# Helper functions
def disp_feature_dynamics(
    expr_df, 
    feature, 
    std_df=None,
    figsize=(6, 2.5)
):
    n_bins = expr_df.shape[1]
    xx = np.arange(n_bins)
    yy = expr_df.loc[feature]

    plt.figure(figsize=figsize)
    if std_df is None:
        plt.scatter(xx, expr_df.loc[feature], s=1, c='k', alpha=.5)
    else:
        plt.plot(xx, yy, linewidth='.5', c='k', linestyle='-.')
        plt.fill_between(xx, yy-std_df.loc[feature], yy+std_df.loc[feature], color='blue', alpha=.1)

    plt.xlabel(r"PV $\rightarrow$ CV bins", fontsize=12)
    plt.ylabel('Expression', fontsize=12)
    plt.title(feature, fontsize=15)
    plt.show()


# Visualization
def summarize_edge_weights(
    adata,  
    ccc_rep='omega', 
    cluster_key='cell_type', 
    cluster_labels=None,
    title='', 
    show_fig=False
):
    """Compute cluster-wise summary of cell-cell interactions"""
    if cluster_labels is None:
        cluster_labels = adata.obs[cluster_key].cat.categories
    per_idx_labels = adata.obs['cell_type'].values
    n_clusters = len(cluster_labels)
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    # Aggregate: for each receiver type, average over its cells
    for i, rtype in enumerate(cluster_labels):
        mask = (per_idx_labels == rtype)
        if mask.sum() > 0:
            mat[i] = adata.obsm[ccc_rep][mask].mean(axis=0)   # sender cell types

    # add omega as an extra sender column
    df = pd.DataFrame(
        mat,
        index=cluster_labels, 
        columns=list(cluster_labels)
    )

    # plot heatmap
    if show_fig:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, cmap="magma", linecolor='gray', linewidth=0.5)
        plt.xlabel("Sender", fontsize=10)
        plt.ylabel("Receiver", fontsize=10)
        plt.title(title, fontsize=20)
        plt.show()

    return df


def plot_incoming_over_t(
    # adata, target_cell_type, scores, 
    # bin_t=None, normalize="none", show=False
    df_wide, target_cell_type, show=True, title='Incoming interactions'
):
    """
    Plot line plot of incoming interactions to a specific receiver cell type
    across trajectory coordinate t, with optional binning.
    Also returns a wide-format dataframe: [t, zone, sender_1, ..., sender_C].
    """
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm

    zone_bounds = []
    for z in np.unique(df_wide['zone']):
        t_min = df_wide.loc[df_wide["zone"] == z, "t"].min()
        t_max = df_wide.loc[df_wide["zone"] == z, "t"].max()
        zone_bounds.append((z, t_min, t_max))

    # ---- plotting ----
    if show:
        plt.figure(figsize=(10, 4))
        ax = sns.lineplot(
            data=df_wide.melt(id_vars=["t", "zone"], var_name="sender", value_name="interaction"),
            x="t", y="interaction", hue="sender",
            estimator="mean", errorbar="se", lw=2
        )

        # add colored zone bands
        ymin, ymax = plt.ylim()
        y_range = ymax - ymin
        band_height = 0.05 * y_range

        cmap = cm.get_cmap("turbo", len(zone_bounds))
        for i, (z, t_min, t_max) in enumerate(zone_bounds):
            color = cmap(i)
            ax.add_patch(Rectangle(
                (t_min, ymin - band_height),
                t_max - t_min,
                band_height,
                color=color, 
                # alpha=0.5
            ))
            ax.text(
                (t_min+t_max)/2, ymin - band_height/2,
                f"Zone {int(z)+1}", ha="center", va="center", 
                fontsize=8, color="white", weight="bold"
            )

        plt.ylim(ymin - band_height, ymax)
        plt.title(f"{title} → {target_cell_type}")
        plt.ylabel("Proportion (Log-scaled)", fontsize=12)
        plt.xlabel(r"Trajectory ($t$)", fontsize=12)
        plt.legend(title="Sender (cell-type)", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    return None

# %%
# TMP: figure plots
# (1). Comparison of CCI vs. Abundance for incoming T-cell along zones
t_cell_abun_df = pd.read_csv('../results/liver/cci/t-cell_incoming_abun.csv', index_col=0)
t_cell_cci_df = pd.read_csv('../results/liver/cci/t-cell_incoming_int.csv', index_col=0)
plot_incoming_over_t(t_cell_abun_df, target_cell_type='T-cells', title='Incoming Abundance')
plot_incoming_over_t(t_cell_cci_df, target_cell_type='T-cells')

# %%
cci_compiled = pickle.load(open('../results/liver/cci/zone_interactions.pkl', 'rb'))

# %%
cell_types = cci_compiled['0'].columns
colors = plot.generate_random_colors(len(cell_types))
palette = dict(zip(cell_types, colors))

for zone in ['0', '1', '2', '3', '4']:
    fig, ax = plot.netVisual_circle(
        cci_compiled[zone], 
        title=f'Zone {int(zone)+1} Interactions',
        edge_width_max=10,
        vertex_size_max=15,
        use_sender_colors=False,
        curve_strength=0.1,
        figsize=(15,11)
    )
    plt.show()

del cell_types, colors, palette
gc.collect()



# %%
# ---------------------------------
#  I. Trajectory analysis
# ---------------------------------
n_latent = 6
n_zones = 5
n_bins = 50
sample_id = 'NIH_F5'

# Binned expression per sample
print('Analyzing {}...'.format(sample_id))
adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

# qzx = np.load('../results/LYNX_xenium_{0}_{1}.npy'.format(n_latent, sample_id))
# qzu = np.load('../results/LYNX_desi_{0}_{1}.npy'.format(n_latent, sample_id))
qzx = np.load('../results/liver/LYNX_xenium_{0}_debug1.npy'.format(n_latent))
qzu = np.load('../results/liver/LYNX_desi_{0}_debug1.npy'.format(n_latent))
    
adata_xenium.obsm['X_z'] = qzx
adata_desi.obsm['X_z'] = qzu

trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z',
    root_marker='DPT',
)

trajectory.compute_trajectory(
    adata_desi, 
    use_rep='X_z',
    root_marker='Taurine '
)

sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)

# Compute discrete zonations
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    abundance_test=True, show=True
)

# Compute feature dynamics along trajectory
# Sorting & binning genes
indices = np.argsort(adata_xenium.obs['t']).values
gexp_df = utils.get_binned_expr(
    adata_xenium.to_df().iloc[indices].T,
    n_bins=n_bins,
)

gamma = utils.get_binned_expr(
    pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()
gexp_df = gexp_df.T
gexp_df['t'] = gamma


del indices, gamma
gc.collect()

# Sorting & binning metabolites
indices = np.argsort(adata_desi.obs['t']).values
mexp_df = utils.get_binned_expr(
    adata_desi.to_df().iloc[indices].T,
    n_bins=n_bins,
)

gamma = utils.get_binned_expr(
    pd.DataFrame(adata_desi.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()

mexp_df = mexp_df.T
mexp_df['t'] = gamma

# Compute phenotype dynamics along the trajectory
celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs['cell_type'], n_bins=n_bins)
# celltype_dynamics_df['t'] = gamma
# celltype_dynamics_df['sample_id'] = sample_id
# celltype_dynamics_df['sex'] = 'M' if 'M' in sample_id else 'F'

del indices, gamma
gc.collect()

# %%
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    abundance_test=True,
    n_zones=5, sample_id=sample_id,
    show=True
)

# %%
# Visualization
# (1). DEGs per zone
sc.pl.rank_genes_groups_matrixplot(
    adata_xenium, n_genes=5, use_raw=False, cmap='RdBu_r'
)

# %%
# (2). phenotype dynamics along the trajectory
celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs['cell_type'], n_bins=n_bins)
plot.disp_celltype_dynamics(celltype_dynamics_df)


# %%
# ----------------------------------
#  II.  Sex-specific joint analysis
# ----------------------------------

# (1). Continuous statistical test: joint analysis of significant features
#  - along PV - CV trajectory
#  - sex-specific

# %%
sample_ids = sorted([
    sample_id for sample_id in os.listdir(xenium_path)
    if os.path.isdir(os.path.join(xenium_path, sample_id))
])


# %%
n_latent = 6
n_zones = 5
n_bins = 20

# Binned expression per sample
gexps = [] 
mexps = []

celltype_dynamics = []

adatas_xenium = []
adatas_desi = []

for sample_id in sample_ids:
    print('Analyzing {}...'.format(sample_id))
    adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')


    qzx = np.load('../results/lynx_hetero_xenium_{0}_{1}.npy'.format(n_latent, sample_id))
    qzu = np.load('../results/lynx_hetero_desi_{0}_{1}.npy'.format(n_latent, sample_id))
    
    adata_xenium.obsm['X_z'] = qzx
    adata_desi.obsm['X_z'] = qzu

    trajectory.compute_trajectory(
        adata_xenium, 
        use_rep='X_z',
        root_marker='DPT',
    )

    trajectory.compute_trajectory(
        adata_desi, 
        use_rep='X_z',
        root_marker='Taurine '
    )

    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

    # Compute feature dynamics along trajectory
    # Sorting & binning genes
    indices = np.argsort(adata_xenium.obs['t']).values
    gexp_df = utils.get_binned_expr(
        adata_xenium.to_df().iloc[indices].T,
        n_bins=n_bins,
    )

    gamma = utils.get_binned_expr(
        pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
        n_bins=n_bins
    ).values.flatten()

    gexp_df = gexp_df.T
    gexp_df['t'] = gamma
    gexp_df['sample_id'] = sample_id
    gexp_df['sex'] = 'M' if 'M' in sample_id else 'F'

    gexps.append(gexp_df)

    # Sorting & binning metabolites
    indices = np.argsort(adata_desi.obs['t']).values
    mexp_df = utils.get_binned_expr(
        adata_desi.to_df().iloc[indices].T,
        n_bins=n_bins,
    )

    gamma = utils.get_binned_expr(
        pd.DataFrame(adata_desi.obs['t'].sort_values()).T,
        n_bins=n_bins
    ).values.flatten()

    mexp_df = mexp_df.T
    mexp_df['t'] = gamma
    mexp_df['sample_id'] = sample_id
    mexp_df['sex'] = 'M' if 'M' in sample_id else 'F'

    mexps.append(mexp_df)

    # Compute phenotype dynamics along the trajectory
    celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs['cell_type'], n_bins=n_bins)
    celltype_dynamics_df['t'] = gamma
    celltype_dynamics_df['sample_id'] = sample_id
    celltype_dynamics_df['sex'] = 'M' if 'M' in sample_id else 'F'
    celltype_dynamics.append(celltype_dynamics_df)

    adatas_xenium.append(adata_xenium)
    adatas_desi.append(adata_desi)

    del adata_xenium, adata_desi, qzx, qzu, gamma, gexp_df, mexp_df
    del sample_id, indices
    gc.collect()


# %%
# Linear-mixed effect models on each feature
all_gexp_df = pd.concat(gexps, axis=0)
all_genes = all_gexp_df.columns[:-3]
fitted_gexp_df, gene_test_assocs = test_assoc.get_test_associations(all_gexp_df)

all_mexp_df = pd.concat(mexps, axis=0)
all_metabolites = all_mexp_df.columns[:-3]
fitted_mexp_df, metabolite_test_assocs = test_assoc.get_test_associations(all_mexp_df)


# ---------------------------------------------
#  LMEs for sex & dynasmics statistical tests
# ---------------------------------------------

# %%
gene_test_assocs[gene_test_assocs['adj-pval.sex'] < .05].index

# %%
# Visualize feature dynamics with significant sex disparity
sex_genes = gene_test_assocs[gene_test_assocs['adj-pval.sex'] < .05].index

print('sex-disparity genes')
print('===================================')
idx = 0
ncols = 4

while idx < len(sex_genes):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_genes):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_gexp_df, feature=sex_genes[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()


# %%
# Load metabolite annotations & update annotations
metabolite_annots_df = pd.read_csv('../data/metabolite_annotations_pos_mode.csv')
metabolite_dict = {
    k: v for k, v in zip(metabolite_annots_df.iloc[:, 0], metabolite_annots_df.iloc[:, 1])
    if not pd.isna(v)
}

# Update test assocs
metabolite_features = []
for feature in metabolite_test_assocs.index:
    if feature in metabolite_dict:
        metabolite_features.append(metabolite_dict[feature])
    else:
        metabolite_features.append(feature)
metabolite_test_assocs.index = metabolite_features

# Update binned expressions
metabolite_features = []
for feature in all_mexp_df.columns:
    if feature in metabolite_dict:
        metabolite_features.append(metabolite_dict[feature])
    else:
        metabolite_features.append(feature)
all_mexp_df.columns = metabolite_features

# for i in range(len(mexps)):
#     col = []
#     for feature in mexps[i].columns:
#         if feature in metabolite_dict:
#             col.append(metabolite_dict[feature])
#         else:
#             col.append(feature)  
#     mexps[i].columns = col
        
del feature, metabolite_dict, metabolite_features, metabolite_annots_df

# %%
sex_metabolites = metabolite_test_assocs[metabolite_test_assocs['adj-pval.sex'] < .05].index
print('sex-disparity metabolites')
print('===================================')
idx = 0
ncols = 4

while idx < len(sex_metabolites):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_metabolites):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_mexp_df, feature=sex_metabolites[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()

# %%
# %% Visualize example features
fig, ax = plt.subplots(figsize=(4, 3))
plot.disp_sex_feature_dynamics(all_gexp_df, feature='HAMP', ax=ax)

# %% 
# Other metabolite visualizations
# DG + TG (hypothesis: enrichment in male + CV)
glycerides = metabolite_test_assocs[
    np.logical_or(
        metabolite_test_assocs.index.str.contains('DG'),
        metabolite_test_assocs.index.str.contains('TG')
    )
].index

# %%
print('Glycerides')
print('===================================')
idx = 0
ncols = 4

while idx < len(glycerides):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(glycerides):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                all_mexp_df, feature=glycerides[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()

# %%
# Visualize indiv. features
fig, ax = plt.subplots(figsize=(4, 3))
plot.disp_sex_feature_dynamics(
    all_mexp_df, feature=glycerides[50], ax=ax
)

# %%
all_mexp_df.head()


# %%
# DG/TG sex-dependent coefficients vs. randomized samples
glycerides_test_assocs = metabolite_test_assocs.loc[glycerides].copy()
glycerides_test_assocs['Category'] = 'glycerides'

random_test_assocs = metabolite_test_assocs.loc[
    np.random.choice(metabolite_test_assocs.index, len(glycerides), replace=False)
].copy()
random_test_assocs['Category'] = 'random'

glycerides_test_assocs = pd.concat((glycerides_test_assocs, random_test_assocs))
glycerides_test_assocs.head()

# %%
glycerides_test_assocs.head()

# %%
from statannotations.Annotator import Annotator

rcParams.update({'font.size': 10})

fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
sns.violinplot(glycerides_test_assocs, x='Category', y='coeff.sex',  linewidth=1.5, palette='seismic', ax=ax)
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xlabel('Metabolite Category')
ax.set_ylabel('Regression coefficients\n (Male > Female)')


pairs = [('glycerides', 'random')]
annotator = Annotator(
    ax, pairs, data=glycerides_test_assocs, x='Category', y='coeff.sex',
)
annotator.configure(test='Mann-Whitney', text_format='full', loc='outside')
annotator.apply_and_annotate()
fig.suptitle('Sex-specific abundance (Glycerides)', fontsize=14, y=1.02)
fig.show()


# %%
# Save sex-dependent & trajectory dependent test statistics

gene_test_assocs.to_csv(os.path.join(outdir, 'gene_test_assocs.csv'), index=True)
    
# Update metabolite +/- annotations
ion_modes = [
    IO.check_ion_mode(
        ion, 
        pos_path='../data/desi/desi_2d/pos/NIH_F1.ome.tif',
        neg_path='../data/desi/desi_2d/neg/NIH_F1.ome.tif',
    )
    for ion in metabolite_test_assocs.index
]
metabolite_test_assocs['+/-'] = ion_modes
metabolite_test_assocs.to_csv(os.path.join(outdir, 'metabolite_test_assocs.csv'), index=True)

# %%
# cell-type dynamics + metabolite group dynamics
# TODO: unify cell-type annotation names
celltype_dynamics[4].columns = [
    'Cholangiocytes + Progenitor', 'Endothelial', 'Fibroblasts',
    'Hepatocytes', 'Kupffer', 'M2', 'Sinusoidal',
    'Smooth Muscle cells', 'T-cells', 't', 'sample_id', 'sex'
]
celltype_dynamics_df = celltype_dynamics[4]

# %%
idx = 0
ncols = 2
cell_types = [
    'Cholangiocytes + Progenitor', 'Endothelial', 'Fibroblasts',
    'Hepatocytes', 'M2', 'Sinusoidal', 'Smooth Muscle cells', 'T-cells'
]

while idx < len(cell_types):
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 2.5))
    for ax in axes:
        if idx >= len(celltype_dynamics):
            ax.axis('off')
        else:
            ax = disp_sex_feature_dynamics(
                celltype_dynamics, sample_ids, feature=cell_types[idx], 
                ax=ax, ylabel='Proportions', show=False
            )
        idx += 1
    plt.show()


# %%
# TMP: remove Kupffer test for now (missing annotation in 1 sample)
for i in range(len(celltype_dynamics)):
    if 'Kupffer' in celltype_dynamics[i].columns:
        celltype_dynamics[i].drop('Kupffer', axis=1, inplace=True)

# %%
all_celltype_dynamics_df = pd.concat(celltype_dynamics, axis=0)
phenotype_test_assocs = test_assoc.get_test_associations(all_celltype_dynamics_df)
phenotype_test_assocs


# %%
# -------------------------
#  Cell-type localization
# -------------------------

import holoviews as hv
from holoviews import opts, dim
hv.extension('bokeh')


# %%
# Save cell-type colocalization along trajectory (circos plot?)
def plot_celltype_interaction(attn_df, amplitude=1):
    r"""Visualize cell-type colocalization / interactions per zone"""
    assert np.array_equal(attn_df.index, attn_df.columns)
    attn_score = attn_df.values
    cell_types = attn_df.columns

    graph = hv.Graph([
        (cell_types[i], cell_types[j], attn_score[i, j])
        for i in range(len(cell_types)-1) for j in range(i+1, len(cell_types))
    ], vdims=['weight'])
    labels = hv.Labels(graph.nodes, ['x', 'y'], 'index')

    graph = graph.opts(
        node_color='index', edge_color=hv.dim('weight')*amplitude, cmap='Category10',
        edge_cmap='Reds', edge_line_width=hv.dim('weight')*amplitude,
    )
    graph = (graph * labels.opts(text_font_size='10pt', text_color='black'))

    return graph


n_latent = 6
n_zones = 5


for sample_id in sample_ids:
    adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

    qzx = np.load('../results/lynx_hetero_xenium_{0}_{1}.npy'.format(n_latent, sample_id))
    qzu = np.load('../results/lynx_hetero_desi_{0}_{1}.npy'.format(n_latent, sample_id))

    adata_xenium.obsm['X_z'] = qzx
    adata_desi.obsm['X_z'] = qzu

    trajectory.compute_trajectory(
        adata_xenium, 
        use_rep='X_z',
        root_marker='DPT',
    )

    trajectory.compute_trajectory(
        adata_desi, 
        use_rep='X_z',
        root_marker='Taurine '
    )

    utils.get_zonations(adata_xenium, n_zones=n_zones)

    # preprocess data
    # sc.pp.normalize_total(adata_xenium)
    # sc.pp.log1p(adata_xenium)

    # Constructing graphs
    colocalize_graphs = []
    cluster_key = 'cell_type'

    for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
        
        adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()

        sq.gr.spatial_neighbors(adata_sub, coord_type='generic', n_neighs=100, radius=50, )
        sq.gr.nhood_enrichment(adata_sub, cluster_key=cluster_key)

        enrich = adata_sub.uns[cluster_key+'_nhood_enrichment']['zscore']
        # enrich[enrich <= 1.65] = 0.
        enrich_df = pd.DataFrame(
            enrich,
            index=adata_sub.obs[cluster_key].cat.categories,
            columns=adata_sub.obs[cluster_key].cat.categories
        )

        sq.pl.nhood_enrichment(adata_sub, cluster_key=cluster_key, cmap='magma')

        graph = plot_celltype_interaction(enrich_df, amplitude=1)
        colocalize_graphs.append(graph)

    del cluster_id, enrich, enrich_df, graph, adata_sub

    # Saving
    holomap = hv.HoloMap({i: graph for i, graph in enumerate(colocalize_graphs)},  kdims='{}\nBin (PV->CV)'.format(sample_id))
    holomap = holomap.opts(
        xaxis=None, yaxis=None, axiswise=True,
        width=500, height=500
    ) 

    hv.save(
        holomap,
        '../results/downstream/cell_types/{}_colocalization'.format(sample_id), fmt='widgets'
    )

    del adata_sub, colocalize_graphs, holomap
    gc.collect()



# %%
# Save zone-specific DEGs per sex (3 / 5)


# %%
# (2). Discrete statistical test: differentially expressed features per zone 

# %%
# Volcano plot: differentially expressed features
def display_volcano(
    df, lfc_label, pval_label,
    p=0.01, lfc=1, s=0.2, 
    figsize=(4, 3), title=None, dpi=300,
    ax=None, show=True
):
    lfc_idx = df.columns.str.contains(lfc_label)
    pval_idx = df.columns.str.contains(pval_label)

    x = df.loc[:, lfc_idx].values.flatten()
    y = df.loc[:, pval_idx].values.flatten()
    
    is_upreg = np.logical_and(x > lfc, y < p)
    is_downreg = np.logical_and(x < -lfc, y < p)
    is_nonsig = np.logical_and(~is_upreg, ~is_downreg)
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x[is_nonsig], -np.log10(y[is_nonsig]), s=s, edgecolors='none', c='gray')
    ax.scatter(x[is_upreg], -np.log10(y[is_upreg]), s=s, edgecolors='none', c='firebrick')
    ax.scatter(x[is_downreg], -np.log10(y[is_downreg]), s=s, edgecolor='none', c='dodgerblue')
    ax.set_xlabel('Log2FC')
    ax.set_ylabel('-log10(adj.pvalue)')

    ax.axvline(x=lfc, c='lightgray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-lfc, c='lightgray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-np.log10(p), c='lightgray', linestyle='--', linewidth=0.5)

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()

    upreg_df = df.loc[is_upreg].copy()
    downreg_df = df.loc[is_downreg].copy()

    for feature, yy, xx in zip(upreg_df.iloc[:5, 0], upreg_df.iloc[:5, -2], upreg_df.iloc[:5, -1]):
        xx -= .01
        yy = -np.log10(yy) + 3
        if (~np.isinf(xx) and ~np.isinf(yy)):
            ax.text(xx, yy, feature, fontsize=5, c='red', weight='bold')

    for feature, yy, xx in zip(downreg_df.iloc[-5:, 0], downreg_df.iloc[-5:, -2], downreg_df.iloc[-5:, -1]):
        xx -= .01
        yy = -np.log10(yy) + 3
        if (~np.isinf(xx) and ~np.isinf(yy)):
            ax.text(xx, yy, feature, fontsize=5, c='blue', weight='bold')
    
    ax.set_title(title, fontsize=10)
    
    if show:
        plt.show()
        return None    
    else:
        return ax, df.loc[is_upreg], df.loc[is_downreg], df

    
# %%
outdir_diff_expr = os.path.join(outdir, 'pathway_' + str(n_zones))
if not os.path.exists(outdir_diff_expr):
    os.makedirs(outdir_diff_expr, exist_ok=True)

for sample_id, adata, adata_desi in zip(sample_ids[1:], adatas_xenium, adatas_desi):
    for i in range(n_zones):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=500)
        ax1, upreg_df, downreg_df, df = display_volcano(
            adata.uns['zones'][str(i)],
            lfc_label='logFC', pval_label='pvals_adj', 
            lfc=.25, p=1e-3, s=1., dpi=500,
            title='{0} (Xenium) - Zone {1}'.format(sample_id, i), 
            ax=ax1, show=False
        )

        # upreg_df.to_csv(
        #     os.path.join(outdir_diff_expr, 'UP_gene_{0}_zone_{1}.csv'.format(sample_id, i)),
        #     index=False
        # )

        # downreg_df.to_csv(
        #     os.path.join(outdir_diff_expr, 'DOWN_gene_{0}_zone_{1}.csv'.format(sample_id, i)),
        #     index=False
        # )

        ax2, upreg_df, downreg_df = display_volcano(
            adata_desi.uns['zones'][str(i)],
            lfc_label='logFC', pval_label='pvals_adj', 
            lfc=.1, p=1e-3, s=1., dpi=500,
            title='{0} (DESI) - Zone {1}'.format(sample_id, i), 
            ax=ax2, show=False
        )

        # upreg_df.to_csv(
        #     os.path.join(outdir_diff_expr, 'UP_metabolite_{0}_zone_{1}.csv'.format(sample_id, i)),
        #     index=False
        # )

        # downreg_df.to_csv(
        #     os.path.join(outdir_diff_expr, 'DOWN_metabolite_{0}_zone_{1}.csv'.format(sample_id, i)),
        #     index=False
        # )

        plt.tight_layout()
        plt.show()

del sample_id, adata, adata_desi, upreg_df #, downreg_df
gc.collect()

# %%
# Pooled analysis by sex (Male vs. Female)
# e.g. 3-zones

diff_expr_path = '../results/downstream/pathway_{}'.format(n_zones)

zone_0_genes = {'F': {}, 'M': {}}
zone_1_genes = {'F': {}, 'M': {}}
zone_2_genes = {'F': {}, 'M': {}}

zone_0_metabolites = {'F': {}, 'M': {}}
zone_1_metabolites = {'F': {}, 'M': {}}
zone_2_metabolites = {'F': {}, 'M': {}}

files = sorted([
    os.path.join(diff_expr_path, f)
    for f in os.listdir(diff_expr_path)
])

for f in files:
    df = pd.read_csv(f)
    sex = f.split('_NIH_')[-1][0]
    features = df.iloc[:, 0]

    if 'gene' in f:
        if 'zone_0' in f:
            for feature in features:
                zone_0_genes[sex][feature] = zone_0_genes[sex].get(feature, 0) + 1
        elif 'zone_1' in f:
            for feature in features:
                zone_1_genes[sex][feature] = zone_1_genes[sex].get(feature, 0) + 1
        else:
            for feature in features:
                zone_2_genes[sex][feature] = zone_2_genes[sex].get(feature, 0) + 1     

    else:
        if 'zone_0' in f:
            for feature in features:
                zone_0_metabolites[sex][feature] = zone_0_metabolites[sex].get(feature, 0) + 1
        elif 'zone_1' in f:
            for feature in features:
                zone_1_metabolites[sex][feature] = zone_1_metabolites[sex].get(feature, 0) + 1
        else:
            for feature in features:
                zone_2_metabolites[sex][feature] = zone_1_metabolites[sex].get(feature, 0) + 1     

    del f, df, sex

# %%
def get_sex_features(summary: dict, meta_annot: str = None):
    """Features 3/5 per sex"""
    male_features = [feature.strip() for feature in summary['M'] if summary['M'][feature] >= 3]
    female_features = [feature.strip() for feature in summary['F'] if summary['F'][feature] >= 3]

    if meta_annot is None:
        return pd.DataFrame(male_features, columns=['#Official']), pd.DataFrame(female_features, columns=['#Official'])
    else:
        male_named_features = []
        for feature in male_features:
            if 'm/z' not in feature:
                male_named_features.append(feature)
            elif feature in meta_annot:
                male_named_features.append(meta_annot[feature])
            else:
                pass
        
        female_named_features = []
        for feature in female_features:
            if 'm/z' not in feature:
                female_named_features.append(feature)
            elif feature in meta_annot:
                female_named_features.append(meta_annot[feature])
            else:
                pass  

        return pd.DataFrame(male_named_features, columns=['#compound']), pd.DataFrame(female_named_features, columns=['#compound'])

# %%
male_genes, female_genes = get_sex_features(zone_2_genes)
male_metabolites, female_metabolites = get_sex_features(zone_2_metabolites, meta_annot=metabolite_annots)

male_genes.to_csv(os.path.join(outdir, 'M_UP_genes_zone_2.csv'), index=False)
female_genes.to_csv(os.path.join(outdir, 'F_UP_genes_zone_2.csv'), index=False)
male_metabolites.to_csv(os.path.join(outdir, 'M_UP_metabolites_zone_2.csv'), index=False)
female_metabolites.to_csv(os.path.join(outdir, 'F_UP_metabolites_zone_2.csv'), index=False)

# %%



