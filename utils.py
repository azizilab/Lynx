import numpy as np
import torch
import networkx as nx

from scipy.stats import zscore
from torchvision import transforms


def norm_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def inv_norm_transform(mean, std):
    return transforms.Compose([
        transforms.Normalize([0., 0., 0.], 1/std),
        transforms.Normalize(-mean, [1., 1., 1.])
    ])


def norm_by_channel(x):
    assert x.ndim == 3, "Image dim needs to be (C, Y, X)"
    x_normed = np.zeros_like(x)
    for i, chan in enumerate(x):
        x_normed[i] = (chan-chan.min())/(chan.max()-chan.min())
    return x_normed


def znorm(v, eps=1e-10):
    """Znorm each feature (dim1)"""
    assert v.ndim == 2, "2D feature matrix required"
    v += eps*np.random.randn(v.shape[0], v.shape[1])
    v_normed = zscore(v)
    assert np.isnan(v_normed).any() == False
    return v_normed


def nx_to_edge_index(G: nx.Graph):
    """Convert networkx graph to Edge-index"""
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    return edge_index

