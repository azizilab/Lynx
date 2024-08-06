import os
import sys

from scipy.stats import zscore
from torchvision import transforms

import pandas as pd 
import matplotlib.pyplot as plt 

import cv2
import numpy as np
import torch
import networkx as nx
import xml.etree.ElementTree as ET

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.filters import gaussian as gaussian_blur
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from collections import OrderedDict
from typing import Optional, Set, List, Dict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def generate_random_colors(n):
    colors = []
    for _ in range(n):
        # Generate a random color
        color = "#{:02x}{:02x}{:02x}".format(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        colors.append(color)
    return colors

  
def norm_by_channel(x):
    x_normed = np.zeros_like(x, dtype=np.float32)
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


def nx_to_edge_attrs(G: nx.Graph):
    """Convert networkx graph to `Edge-Index` & `Edge_Weight`"""
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_weight = None
    if 'weight' in G.edges[next(iter(G.edges))]:
        weight = [data['weight'] for _, _, data in G.edges(data=True)]
        edge_weight = torch.tensor(weight, dtype=torch.float)
    return edge_index, edge_weight


def binary_concrete(p, temp=1):
    """Sample from binary concrete distribution"""
    u = torch.rand_like(p)
    l = torch.log(u / (1-u))
    b = torch.sigmoid((torch.logit(p) + l) / temp)
    return b


# -----------------
#  Autocorrection
# -----------------
  
def apply_otsu_threshold(array):
    thresh = threshold_otsu(array)
    return array > thresh


def apply_AF_threshold(array, percentile=99.5):
    percentile_value = np.percentile(array, percentile)
    return array > percentile_value


# ---------------------------------------
# Extract features from high-dim images
# ---------------------------------------

def get_binned_feature(feature, nbins):
    """Get binned expressions of a specific feature"""
    step = len(feature) // nbins
    binned_means = np.zeros(nbins)
    binned_stds = np.zeros(nbins)
    for i, idx in enumerate(range(0, len(feature), step)):
        if i >= nbins:
            break
        binned_means[i] = feature[idx:idx+step].mean()
        binned_stds[i] = feature[idx:idx+step].std()

    return binned_means, binned_stds


def get_binned_features(features, nbins):
    """Get binned expressions over features for smooth visualization"""
    means = np.zeros((nbins, features.shape[1]))
    stds = np.zeros((nbins, features.shape[1]))
    step = features.shape[0] // nbins
    for i, idx in enumerate(range(0, features.shape[0], step)):
        if i >= nbins:
            break
        means[i] = features[idx:idx+step, :].mean(0)
        stds[i] = features[idx:idx+step, :].std(0)
    return means, stds


def sort_binned_features(features, nbins):
    """
    Get labels sorted based on their argmax location 
    (first to last) along the zonation trajectory

    e.g. expr('A'): [2, 1, 0, 0]; expr('B'): [1, 2, 0, 0]; expr('C'): [0, 0, 1, 0] ==> 
        [expr('A'), expr('B'), expr('C')]
    """
    means, stds = get_binned_features(features, nbins=nbins)   # dim: [#bins, C]
    means, stds = means.T, stds.T
    indices = np.argsort(means.argmax(1))

    sorted_means, sorted_stds = means[indices], stds[indices]
    return sorted_means, sorted_stds, indices


def infer_zones(U, nbins=10, verbose=False):
    """
    Create discretized bins (1,2,...,n) from inferred trajectory
    """    
    qs = np.quantile(U, np.linspace(0, 1, nbins+1))
    if verbose:
        print('Quantile:', qs)
        
    zone = np.zeros_like(U, dtype=np.int32)
    for i, q in enumerate(qs[:-1]):
        zone[U >= q] = i

    return zone
 
def get_roi_mask(img: np.ndarray, 
                 sigma: float = 5.,
                 min_area: float = 0.):
    """Compute binary matrix for ROI selection without background pixels """
    img_blurred = img.copy() if img.ndim == 2 \
                  else img.mean(0)  # dim: [Y, X] or [C, Y, X]
    img_blurred = gaussian_blur(img_blurred, sigma=sigma)
    mask = apply_otsu_threshold(img_blurred)
    if min_area > 0:
        mask = remove_holes(mask, min_area)
    return mask


def remove_holes(roi, min_area):
    """ Remove holes & FP lslands in binary ROI mask"""
    roi_filtered = roi.copy().astype(np.uint8)
    roi_labeled, n_features = ndi.label(roi)
    
    for i in range(1, n_features+1):
        if (roi_labeled == i).sum() < min_area:
            roi_filtered[roi_labeled == i] = 0
            
    return ndi.binary_fill_holes(roi_filtered).astype(np.uint8)


def feature_to_img(feature_mat: np.ndarray, mask: np.ndarray):
    """Convert (Pixel x Channel) expression back to spatial image w/ ROI mask"""
    assert mask.ndim == 2, "Invalid mask dimension {}".format(mask.ndim)
    ndimy, ndimx = mask.shape
    img = np.zeros((ndimy, ndimx), dtype=feature_mat.dtype)
    img[np.nonzero(mask)] = feature_mat
    return img


def create_vein_mask(src_chan, sink_chan, q=0.05, sigma=1.5):    
    """Binarize Source & Sink to obtain CV / PV approximation""" 
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

