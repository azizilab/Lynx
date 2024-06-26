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
from IO import *


def generate_random_colors(n):
    colors = []
    for _ in range(n):
        # Generate a random color
        color = "#{:02x}{:02x}{:02x}".format(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        colors.append(color)
    return colors


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

def norm_features(v):
    """Norm each feature (dim1) to [0, 1]"""
    v_normed = np.zeros_like(v)
    for j, feature in enumerate(v.T):
        v_normed[:, j] = (feature-feature.min()) / (feature.max()-feature.min())
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


def PD_approx(cov, eps=1e-6):
    try:
        np.linalg.cholesky(cov)
        return cov
    except np.linalg.LinAlgError:
        eigvals, Q = np.linalg.eigh(cov)
        L_prime = np.diag(
            np.vectorize(lambda x: max(x, eps))(eigvals)
        )
        return Q @ L_prime @ Q.T


# -----------------
#  Clustering
# -----------------

def max_dendrogram_depth(linkage_matrix):
    """Traverse through `Z` and return the root depth"""
    num_samples = linkage_matrix.shape[0] + 1
    depths = np.zeros(num_samples * 2 - 1)
    
    for i in range(linkage_matrix.shape[0]):
        left = int(linkage_matrix[i, 0])
        right = int(linkage_matrix[i, 1])
        depths[num_samples + i] = max(depths[left], depths[right]) + 1
        
    return int(depths[-1])


def max_dendrogram_height(linkage_matrix):
    """Compute the maximum height from the linkage matrix"""
    heights = linkage_matrix[:, 2]
    return np.max(heights)

def get_dendrogram_cluster(Z, ratio=0.2):
    """Tree-cut for cluster construction"""
    cutoff_distance = ratio*max_dendrogram_height(Z)
    return fcluster(Z, cutoff_distance, criterion='distance')


# -----------------
#  Autocorrection
# -----------------
  
def apply_otsu_threshold(array):
    thresh = threshold_otsu(array)
    return array > thresh


def apply_AF_threshold(array, percentile=99.5):
    percentile_value = np.percentile(array, percentile)
    return array > percentile_value


def otsu_correction(input_dir, output_path):
    """This function corrects for AF using otsu's thresholding:
    input: input directory and output directory paths.
    output: ome.tif AF corrected images with same file names in the output directory.
    Should be applied only to images with no gradient issues"""
    
    annot_imgs, filenames = load_annot_tiffs(input_dir, ext='ome.tif')
    af_data = pd.read_csv('/home/jz3553_columbia_edu/liver3d/autof.csv')
    af_data['Channel'] = af_data['Channel'].astype(str)
    af_data['AF'] = af_data['AF'].astype(str)
    autofluorescent_keys = ['Sample AF_01', 'Sample AF_02', 'Sample AF_03', 'Sample AF_04']

    for image, filename in zip(annot_imgs, filenames):
        for key in image:
            if key in autofluorescent_keys:
                kernel = np.ones((5,5), np.uint8)
                image[key] = cv2.dilate(apply_otsu_threshold(image[key]).astype(np.uint8) * 255, kernel, iterations=1) > 0

        for index, row in af_data.iterrows():
            channel = row['Channel']
            af_channel = row['AF']
            if pd.isna(af_channel) or af_channel not in image:
                continue

            channel_array = image.get(channel, None)
            af_channel_array = image.get(af_channel, None)

            if channel_array is not None and af_channel_array is not None:
                channel_array[af_channel_array] = 0
                image[channel] = channel_array

        dict_processed = {filename: image}
        save_annot_tiffs(dict_processed, output_path, verbose=False)

        
def manual_correction(input_dir, output_path):
    """This function corrects for AF using manual thresholding set at adjustable percentile:
    input: input directory and output directory paths.
    output: ome.tif AF corrected images with same file names in the output directory.
    Should be applied only to images that have very high intensity spots that cannot be removed from otsu."""
    
    annot_imgs, filenames = load_annot_tiffs(input_dir, ext='ome.tif')
    af_data = pd.read_csv('/home/jz3553_columbia_edu/liver3d/autof.csv')
    af_data['Channel'] = af_data['Channel'].astype(str)
    af_data['AF'] = af_data['AF'].astype(str)
    autofluorescent_keys = ['Sample AF_01', 'Sample AF_02', 'Sample AF_03', 'Sample AF_04']

    for image, filename in zip(annot_imgs, filenames):
        for key in image:
            if key in autofluorescent_keys:
                kernel = np.ones((5,5), np.uint8)
                image[key] = cv2.dilate(apply_AF_threshold(image[key]).astype(np.uint8) * 255, kernel, iterations=1) > 0
        
        for index, row in af_data.iterrows():
            channel = row['Channel']
            af_channel = row['AF']
            if pd.isna(af_channel) or af_channel not in image:
                continue

            channel_array = image.get(channel, None)
            af_channel_array = image.get(af_channel, None)

            if channel_array is not None and af_channel_array is not None:
                channel_array[af_channel_array] = 0
                image[channel] = channel_array

        dict_processed = {filename: image}
        save_annot_tiffs(dict_processed, output_path, verbose=False)

        
def bcatenin_correction(input_dir, output_path):
    """This function corrects for beta-catenine channel bleeding into the subsequent ASS1 channel:
    input: input directory and output directory paths.
    output: ome.tif AF corrected images with same file names in the output directory.
    Should be applied only to images with gradient issues"""
    
    annot_imgs, filenames = load_annot_tiffs(input_dir, ext='ome.tif')
    af_data = pd.read_csv('/home/jz3553_columbia_edu/liver3d/autof.csv')
    af_data['Channel'] = af_data['Channel'].astype(str)
    af_data['AF'] = af_data['AF'].astype(str)
    autofluorescent_keys = ['Sample AF_01', 'Sample AF_02', 'Sample AF_03', 'Sample AF_04']

    for image, filename in zip(annot_imgs, filenames):
        for key in image:
            if key in autofluorescent_keys:
                kernel = np.ones((5,5), np.uint8)
                image[key] = cv2.dilate(apply_AF_threshold(image[key], 99.5).astype(np.uint8) * 255, kernel, iterations=1) > 0

        for index, row in af_data.iterrows():
            channel = row['Channel']
            af_channel = row['AF']
            if pd.isna(af_channel) or af_channel not in image:
                continue

            channel_array = image.get(channel, None)
            af_channel_array = image.get(af_channel, None)

            if channel_array is not None and af_channel_array is not None:
                channel_array[af_channel_array] = 0
                image[channel] = channel_array

        # Additional step for B-catenin correction
        if "ASS1 PE" in image and "B-catenin-AF 488" in image:
            image["ASS1 PE"][image["B-catenin-AF 488"]] = 0

        dict_processed = {filename: image}
        save_annot_tiffs(dict_processed, output_path, verbose=False)


# ---------------------------------------
# Extract features from high-dim images
# ---------------------------------------

def get_desi_features(desi_img, coords):
    n_cells = len(coords)
    n_features = len(desi_img)
    features = np.zeros((n_cells, n_features), dtype=np.float32)

    for j, chan in enumerate(desi_img):
        features[:, j] = chan[tuple(coords.T)]
    return features


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


def trim_fov(img: np.ndarray,
             ylim: tuple = None, xlim: tuple = None,
             radius: int = None , channel_axis: int = 0):
    """Create trimmed FOV image stacks"""
    assert img.ndim == 3, "Requires multi-channel input image"
    assert channel_axis == 0 or channel_axis == 2, \
        "Requires dimension ordering as [C, Y, X] or [Y, X, C]"
    raw = img.copy() if channel_axis == 0 else img.transpose(2,0,1)
    ny, nx = raw.shape[-2:]
    if isinstance(ylim, tuple) and isinstance(xlim, tuple):
        assert 0 <= ylim[0] < ylim[1] < ny and 0 <= xlim[0] < xlim[1] < nx, \
            "Invalid trimming ROI range"
        ylow, yhigh = ylim
        xlow, xhigh = xlim
    else:
        radius = min(radius, ny//2, nx//2)
        ylow, yhigh = ny//2 - radius, ny//2 + radius
        xlow, xhigh = nx//2 - radius, nx//2 + radius
    trimmed = raw[:, ylow:yhigh, xlow:xhigh] if channel_axis == 0 \
              else raw[:, ylow:yhigh, xlow:xhigh].transpose(1,2,0)
    return trimmed
