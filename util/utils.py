import os
import sys

from typing import Optional, Set, List, Dict
from IPython.display import display

import pandas as pd 
import matplotlib.pyplot as plt 

import torch
import numpy as np
import scanpy as sc
import squidpy as sq
import networkx as nx
import scFates as scf

from scipy import ndimage as ndi
from scipy.stats import zscore
from scipy.optimize import minimize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian as gaussian_blur
from skimage.morphology import binary_erosion, disk
from sklearn.decomposition import FastICA
from torch_geometric import utils as pyg_utils

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from models.module import GPCALayer


def generate_random_colors(n):
    colors = []
    for _ in range(n):
        # Generate a random color
        color = "#{:02x}{:02x}{:02x}".format(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        colors.append(color)
    return colors

  
# ---------------------------------------
# Preprocessing
# ---------------------------------------
def norm_by_channel(x):
    x_normed = np.zeros_like(x, dtype=np.float32)
    for i, chan in enumerate(x):
        x_normed[i] = (chan-chan.min())/(chan.max()-chan.min())
    return x_normed


def znorm(v, eps=1e-10):
    r"""Znorm each feature (dim1)"""
    assert v.ndim == 2, "2D feature matrix required"
    v += eps*np.random.randn(v.shape[0], v.shape[1])
    v_normed = zscore(v)
    assert np.isnan(v_normed).any() == False
    return v_normed


def pnorm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def nx_to_edge_attrs(G: nx.Graph):
    r"""Convert networkx graph to `Edge-Index` & `Edge_Weight`"""
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_weight = None
    if 'weight' in G.edges[next(iter(G.edges))]:
        weight = [data['weight'] for _, _, data in G.edges(data=True)]
        edge_weight = torch.tensor(weight, dtype=torch.float)
    return edge_index, edge_weight


def get_principal_components(
    adata, 
    n_components, 
    k=30, 
    graph_regularize=False,
    alpha=1.0,
    verbose=False
):
    r"""
    Dimension reduction w/ (graph-regularized) PCA
    """
    if graph_regularize:
        coords = adata.obs[['y_centroid', 'x_centroid']].copy().to_numpy()
        G = construct_graph(coords, k=k, weighted=False)  
        edge_index = pyg_utils.from_networkx(G).edge_index
        model = GPCALayer(
            c_in=adata.X.shape[-1], c_out=n_components, alpha=alpha,
            center=True, init_weight=True, ortho_weight=True
        )
        U_gpca = model(torch.tensor(adata.X).float(), edge_index)
        adata.obsm['X_pca'] = U_gpca.detach().cpu().numpy()
    else:
        sc.pp.pca(adata, n_components)
        if verbose:
            ev = adata.uns['pca']['variance_ratio'].sum()
            print('{0} PCs have total EV ratio={1}'.format(n_components, ev))
    return None


def get_indep_components(x, n_components):
    r"""
    Compute the linear operator W (n_components, n_features)
    for independent sources 
    """
    # l = x.sum(axis=-1, keepdims=True) + 1e-7
    # x = np.log1p(x / l * np.median(l))
    transformer = FastICA(n_components=n_components, random_state=0)
    return transformer.fit(x).components_


def get_highly_variable_metabolites(
    adata,
    n_neighbors=30,
    cutoff=.1,
    n_features=None
):
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        transformation=False
    )
    hvfs = None  # High-variable features
    if n_features is not None:
        hvfs = adata.uns['moranI']['I'][:n_features].index
    else:
        hvfs = adata.uns['moranI']['I'][
            adata.uns['moranI']['I'] > cutoff
        ].index
    return hvfs


def infer_zones(U, n_bins=10, verbose=False):
    r"""
    Create discretized bins (1,2,...,n) from inferred trajectory
    """    
    qs = np.quantile(U, np.linspace(0, 1, n_bins+1))
    if verbose:
        print('Quantile:', qs)
        
    zone = np.zeros_like(U, dtype=np.int32)
    for i, q in enumerate(qs[:-1]):
        zone[U >= q] = i

    return zone
 
 
def get_roi_mask(
    img: np.ndarray, 
    sigma: float = 5.,
    erode_pixel=5,
    min_area: float = 0.
):
    r"""
    Compute binary matrix for ROI by filtering out 
    background & boundary artifact pixels
    """

    def __apply_otsu_threshold(array):
        thresh = threshold_otsu(array)
        return array > thresh

    # `img` dim: [Y, X] or [C, Y, X]
    img_blurred = img.copy() if img.ndim == 2 \
                  else img.mean(0)  

    if sigma == 0. and erode_pixel == 0:
        return np.ones_like(img_blurred, dtype=bool)

    img_blurred = gaussian_blur(img_blurred, sigma=sigma)
    roi_mask = __apply_otsu_threshold(img_blurred)
    if min_area > 0:
        roi_mask = remove_holes(roi_mask, min_area)

    roi_mask = binary_erosion(
        image=roi_mask,
        footprint=disk(radius=erode_pixel)
    )
    return roi_mask


def remove_holes(roi, min_area):
    r""" Remove holes & FP lslands in binary ROI mask"""
    roi_filtered = roi.copy().astype(np.uint8)
    roi_labeled, n_features = ndi.label(roi)
    
    for i in range(1, n_features+1):
        if (roi_labeled == i).sum() < min_area:
            roi_filtered[roi_labeled == i] = 0
            
    return ndi.binary_fill_holes(roi_filtered).astype(np.uint8)


def create_vein_mask(src_chan, sink_chan, q=0.05, sigma=1.5):    
    r"""Binarize Source & Sink to obtain CV / PV approximation""" 
    src_blur = gaussian_blur(src_chan, sigma=sigma)
    thresh = np.quantile(src_blur, 1-q)
    src_prior = (src_chan > thresh).astype(np.uint8)

    sink_blur = gaussian_blur(sink_chan, sigma=sigma)
    thresh = np.quantile(sink_blur, 1-q)
    sink_prior = (sink_chan > thresh).astype(np.uint8)

    u_prior = np.zeros_like(src_chan, dtype=np.int8)
    u_prior[np.logical_and(src_prior == 0, sink_prior == 1)] = 0
    u_prior[np.logical_and(src_prior == 1, sink_prior == 0)] = 1
    return u_prior


# ------------------
#  Post-processing
# ------------------

# --------------------------------------------
# Sorting / binning features along zonations
# --------------------------------------------
def get_binned_expr(
    expr_df, 
    n_bins, 
    std=False, 
    scale=False
):
    r"""Smooth P x N (feature-first) matrix => P x K bins with sliding-window average
    - For computing trajectory dynamics (adata.obs['t']), return `scaled` values to [0, 1]
    - For computing feature expressions, return log-normalized values
    """
    features = expr_df.index
    data = expr_df.values
    expr_proj = np.array_split(data, n_bins, axis=-1)  # dim: [K, P, bin_width]
    
    mean_expr_df = pd.DataFrame(
        np.array([s.mean(-1) for s in expr_proj]).T, 
        index=features
    )
    std_expr_df = pd.DataFrame(
        np.array([s.std(-1) for s in expr_proj]).T,
        index=features
    )

    if scale:
        mean_expr_df = mean_expr_df.apply(
            lambda x: (x-x.min())/(x.max()-x.min()),
            axis=1
        )        

    if std:
        # return mean & std expressions
        return mean_expr_df, std_expr_df
    else:
        return mean_expr_df
        

def sort_fitted_expr(adata):
    r"""Sort expression by both cells (along pseudotime) & 
    features (along peak expression location across pseudotime)
    """
    assert 't' in adata.obs_keys(), \
        "Please run trajectory inference first"
    assert 'fitted' in adata.layers.keys(), \
        "Please fit expressions along the trajectory first"
    
    sorted_cells = adata.obs['t'].sort_values().index
    sorted_genes = scf.pl.trends(
        adata, features=adata.var_names,
        highlight_features='fdr', ordering='max',
        plot_emb=False, show=False, return_genes=True
    )

    # Sort fitted expr by both cell & gene orderings (along the trajectory)
    expr_df = pd.DataFrame(
        adata.layers['fitted'],
        index=adata.obs_names,
        columns=adata.var_names,
    )
    fitted_expr_df = expr_df.loc[sorted_cells, sorted_genes]
    return fitted_expr_df


# --------------------------------------------
#   Zonation & dynamics along the trajectory
# --------------------------------------------
def get_celltype_dynamics(adata, annots, window_size=1000):
    r"""
    Compute cell-type dynamics along the binned trajectory (sliding window)
    """
    assert 't' in adata.obs.columns, \
        "Please infer zonation trajectory first"

    annots = annots.loc[adata.obs_names]
    annots = annots.loc[adata.obs['t'].sort_values().index]    
    
    cell_types = [cell_type for cell_type in np.unique(annots)
              if cell_type != 'Other' and cell_type != 'Unknown']
    window_size = min(window_size, adata.shape[0]//2)
    n_cell_types = len(cell_types)
    n_bins = annots.shape[0] // window_size 
    if annots.shape[0] % window_size != 0:
        n_bins += 1
    dynamics = np.zeros((n_bins, n_cell_types))  # Column: indiv. cell types
        
    idxl = 0
    for i in range(n_bins):
        idxr = annots.shape[0] if i == n_bins-1 else idxl+window_size
        summary = annots[idxl:idxr].value_counts()[cell_types]
        dynamics[i] = (summary / summary.sum()).values
        idxl += window_size
    
    return pd.DataFrame(dynamics, columns=cell_types)


def piecewise_linear_fit(gamma, k, show=False):
    r"""Piesewise linear regression on smoothed trajectory 
    (`gamma`) into discretize into k zones

    Reference: 
    - https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
    """
    X = np.arange(len(gamma))
    Y = np.array(gamma)

    xmin = X.min()
    xmax = X.max()

    seg = np.full(k-1, (xmax - xmin) / k)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 1e-2].mean() for x in px_init])

    def func(p):
        seg = p[:k - 1]
        py = p[k - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    px, py = func(r.x)
    
    if show:
        plt.figure(figsize=(4, 3), dpi=200)
        plt.plot(X, Y)
        plt.plot(px, py, '-or', label='Piecewise regression')
        plt.xlabel('Ordered cells (PV -> CV)')
        plt.ylabel(r"Gradient $\gamma$")
        
        plt.ticklabel_format(style='sci', axis='x')
        plt.legend()
        plt.show()

    return py


def get_zonations(
    adata, 
    n_zones: int = 3, 
    n_bins: int = None,
    show: bool = False
):
    r"""
    Discretize trajectory gradient assignment via piecewise linear regression
    """
    # # Piecewise linear regression, retrieve gradient cutoffs 
    # if n_bins is None:
    #     cutoffs = piecewise_linear_fit(
    #         adata.obs['t'].sort_values().values, 
    #         k=n_zones, show=show
    #     )
    # else:
    #     # Smoothed, sorted gradients
    #     gamma = get_binned_expr(
    #         pd.DataFrame(adata.obs['t'].sort_values()).T,
    #         n_bins=n_bins,
    #         scale=True
    #     ).values.squeeze() 

    #     cutoffs = piecewise_linear_fit(gamma, k=n_zones, show=show)

    # Cutoff by percentile
    thresholds = np.linspace(0, 1, n_zones+1)
    cutoffs = np.quantile(adata.obs['t'].sort_values().values, thresholds[1:-1])
    cutoffs = np.insert(cutoffs, 0, 0)
    cutoffs = np.insert(cutoffs, len(cutoffs), 1)

    # Zonation assignment
    milestones = np.empty_like(adata.obs['t'], dtype=np.uint8)

    for i in range(len(cutoffs)-1):
        mask = np.logical_and(
            adata.obs['t'] >= cutoffs[i],
            adata.obs['t'] < cutoffs[i+1]
        )
        milestones[mask] = i

    milestones[adata.obs['t'] < cutoffs[0]] = 0
    milestones[adata.obs['t'] >= cutoffs[-1]] = n_zones - 1

    if 'milestones_colors' in adata.uns_keys():
        adata.uns.pop('milestones_colors')

    adata.obs['milestones'] = milestones
    adata.obs['milestones'] = adata.obs['milestones'].astype('category')

    if show:
        display(adata.obs['milestones'].value_counts())

    return None


def get_zonation_features(
    adata, 
    adata_desi,
    n_zones,
    sample_id='',
    n_bins=None,
    show=True
):
    r"""Compute zonation (discrete) enriched features"""

    def _get_DE_features(adata, zone_label, feature_name='name'):
        df = sc.get.rank_genes_groups_df(adata, group=zone_label)
        df = df.sort_values('scores', ascending=False).reset_index(drop=True)

        df = df.loc[:, ['names', 'scores', 'pvals_adj', 'logfoldchanges']]
        df.columns = [feature_name, 'TS', 'pvals_adj', 'logFC']

        adata.uns['zones'][str(zone_label)] = df
        adata.uns['zones']['names'][str(zone_label)] = df.iloc[:, 0].values
        adata.uns['zones']['scores'][str(zone_label)] = df.iloc[:, 1].values     

        return None
    
    def _get_matrixplot(adata, title=None):
        markers = {}
        repeats = set()
        for zone_label in np.unique(adata.obs.milestones):
            zone_markers = adata.uns['zones'][str(zone_label)].iloc[:10, 0].values
            markers['Zone '+str(zone_label)] = np.setdiff1d(zone_markers, list(repeats))
            repeats |= set(zone_markers)

        sc.pl.matrixplot(
            adata, markers, groupby='milestones', cmap='RdBu_r',
            standard_scale='var', title=title
        )

        return None

    # Fit trajectory with segmented regression
    get_zonations(adata, n_zones=n_zones, n_bins=n_bins, show=show)
    adata_desi.obs['milestones'] = adata.obs['milestones'].copy()

    # post-hoc differential abundance test
    adata.uns['zones'] = {'names': {}, 'scores': {}}
    adata_desi.uns['zones'] = {'names': {}, 'scores': {}}
    zone_labels = np.unique(adata.obs['milestones'])

    for i, label in enumerate(zone_labels):
        # Only compare target w/ adjacent zones
        # idxl, idxr = max(i-1, 0), min(i+2, len(zone_labels))
        # groups = list(zone_labels[idxl:idxr])

        sc.tl.rank_genes_groups(
            adata, groupby='milestones', # groups=groups,
            method='wilcoxon'
        )
        sc.tl.rank_genes_groups(
            adata_desi, groupby='milestones', # groups=groups,
            method='t-test'
        )
        _get_DE_features(adata, str(label), feature_name='gene')
        _get_DE_features(adata_desi, str(label), feature_name='m.z')
        
    if show:
        group_names = [str(l) for l in zone_labels]
        adata.uns['zones']['params'] = adata.uns['rank_genes_groups']['params']
        adata_desi.uns['zones']['params'] = adata_desi.uns['rank_genes_groups']['params']

        sq.pl.spatial_scatter(
            adata, color='milestones', img=False, size=20,
            title='Zonations ({})'.format(sample_id)
        )

        _get_matrixplot(adata, title='Transcripts ({})'.format(sample_id))
        sc.pl.rank_genes_groups(
            adata, key='zones', groups=group_names, n_genes=10, 
            fontsize=15, ncols=3, sharey=False,
        )

        _get_matrixplot(adata_desi, title='Metabolites ({})'.format(sample_id))
        sc.pl.rank_genes_groups(
            adata_desi, key='zones', groups=group_names, n_genes=10, 
            fontsize=15, ncols=3,sharey=False,
        )

        del adata.uns['zones']['params']

    del adata.uns['zones']['names']
    del adata.uns['zones']['scores']
    del adata.uns['rank_genes_groups']

    return None
