import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_binned_features


def generate_random_colors(n):
    random_colors = []
    for _ in range(n):
        # Generate random RGB values
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        # Convert RGB values to hexadecimal color code
        color_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
        random_colors.append(color_code)
    return random_colors


def disp_chans(img, title=None, ncols=4, cmap='magma'):
    """Display single-channel aligned images"""
    depth = len(img)
    nrows = depth // ncols if depth % ncols == 0 else depth // ncols + 1
    
    idx = 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3.2*nrows))
    for r in range(nrows):
        for c in range(ncols):
            if idx >= depth:
                axes[r, c].axis('off')
                continue
            axes[r, c].imshow(img[idx], cmap=cmap)
            idx += 1
            
    fig.tight_layout()
    fig.suptitle(title, y=1.01)
    plt.show()


def disp_graph_overlaps(Gs, labels, figsize,
                        node_size=5, edge_width=1,
                        title=None):
    colors = generate_random_colors(len(Gs))
    fig, ax = plt.subplots(figsize=figsize)
    for c, lbl, G in zip(colors, labels, Gs):
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos, node_color=c, node_size=node_size, label=lbl, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=c, width=edge_width, ax=ax)

    plt.legend()
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.show() 


def disp_network(img, G, figsize=None,
                 node_size=5, edge_width=1, fontsize=20,
                 title=None):
    pos = {n: G.nodes[n]['pos'] for n in G.nodes}
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='magma', alpha=0.5)
    ax.axis('off')
    nx.draw_networkx_nodes(G, pos, node_color='yellow', node_size=node_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='lime', width=edge_width, ax=ax)
    plt.tight_layout()
    plt.title(title, fontsize=fontsize)
    

def disp_network_embedding(img, G, embedding, figsize=None, 
                           alpha=0.5, img_vmax=64, 
                           node_size=5, edge_width=1,
                           fontsize=20, title=None):
    pos = {n: G.nodes[n]['pos'] for n in G.nodes}
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='magma', alpha=alpha, vmax=img_vmax)
    im = nx.draw_networkx_nodes(G, pos, node_color=embedding, node_size=node_size, cmap='jet', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_width, ax=ax)
    ax.axis('off')    
    plt.colorbar(im, ax=ax, fraction=0.02, pad=1e-4)
    plt.tight_layout()
    plt.title(title, fontsize=fontsize, y=0.95)
    plt.show()


def disp_corr_features(features, labels=None, titles=None, ncols=4):
    n_slices = len(features)
    nrows = n_slices // ncols if n_slices % ncols == 0 else n_slices // ncols + 1

    idx = 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3*nrows), dpi=200)
    for r in range(nrows):
        for c in range(ncols):
            if idx >= n_slices:
                axes[r, c].axis('off')
                continue
            corr = np.corrcoef(features[idx].T)
            sns.heatmap(corr, mask=np.triu(corr),
                        xticklabels=labels, yticklabels=labels,
                        vmin=-0.3, vmax=0.3, square=True, 
                        ax=axes[r, c], cmap='coolwarm')
            
            if titles is not None:
                axes[r, c].set_title(titles[idx])
            idx += 1
    plt.tight_layout()
    plt.suptitle('Feature correlations', fontsize=30, y=1.02)
    plt.show()
    return None


def disp_gradient(feature_means, feature_stds,
                  figsize=(10, 3), dpi=200,
                  vmin=0, vmax=1,
                  title=''):
    """
    Display expressions of a single feature along the trajectory
    """
    xx = np.linspace(0, 1, len(feature_means))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(xx, feature_means, 'b-.', marker='o', linewidth=0.5, markersize=0.7, label='mean')
    ax.fill_between(xx, feature_means-feature_stds, feature_means+feature_stds, alpha=0.2, label='Uncertainty')
    ax.legend()
    ax.set_title(title)

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylim([vmin, vmax])

    ax.set_xlabel('Trajectory')
    ax.set_ylabel('Smoothed expression')

    plt.show()


def disp_gradients(sorted_features, labels,
                   nbins=10, title='', 
                   cluster_ions=True):
    """
    Display feature expressions (binned, sorted) of all cells along the trajectory
    """
    features, _ = get_binned_features(sorted_features, nbins=nbins)  # (coord x feature)
    features_df = pd.DataFrame(features.T, index=labels)

    g = sns.clustermap(features_df, method='ward',
                       row_cluster=cluster_ions, col_cluster=False, 
                       cmap='coolwarm', figsize=(8, 8))
    
    ax = g.ax_heatmap
    
    ax.set_xlabel('Trajectory', fontsize=15)
    ax.set_ylabel('Feature', fontsize=15)
    ax.set_title('Gradients (# bins={0})\n {1}'.format(nbins, title), fontsize=20)
    plt.show()  


def disp_spatial_latent(adata, latent, dim=0, cmap='turbo', vmax=None):
    assert adata.shape[0] == latent.shape[0], \
        "Inconsistent # samples btw inference & dataset"
    assert 0 <= dim < latent.shape[1], \
        "Cluster dim k should be specified btw 0-{}".format(latent.shape[1])

    adata.obs['Z'] = latent[:, dim]
    sq.pl.spatial_scatter(adata, color='Z', vmax=vmax, cmap=cmap, title='Z'+str(dim), size=20, img=False)
    adata.obs.drop('Z', axis=1, inplace=True)
    
    return None


def disp_spatial_latents(adata, latent, ncols=3, cmap='turbo', vmax=None):
    assert adata.shape[0] == latent.shape[0], \
        "Inconsistent # samples btw inference & dataset"
    labels = ['Z'+str(i) for i in range(latent.shape[1])]
    for label, z_k in zip(labels, latent.T):
        adata.obs[label] = z_k
    sq.pl.spatial_scatter(adata, color=labels, vmax=vmax, cmap=cmap, size=20, img=False, ncols=ncols)
    adata.obs.drop(labels, axis=1, inplace=True)
    
    return None