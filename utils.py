import numpy as np
import torch
import networkx as nx

from scipy.stats import zscore
from torchvision import transforms

import sys
import cv2
import pandas as pd 
import matplotlib.pyplot as plt 
import tifffile
import os
import gcsfs
import xml.etree.ElementTree as ET
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian as gaussian_blur
from collections import OrderedDict
from typing import Optional, Set, List, Dict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from IO import *

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

  
def apply_otsu_threshold(array):
    thresh = threshold_otsu(array)
    return array > thresh

def apply_AF_threshold(array, percentile=99.5):
    percentile_value = np.percentile(array, percentile)
    return array > percentile_value

def apply_BC_threshold(array, percentile=97.5):
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
            elif key == "B-catenin-AF 488":
                kernel = np.ones((5,5), np.uint8)
                image[key] = cv2.dilate(apply_BC_threshold(image[key], 97.5).astype(np.uint8) * 255, kernel, iterations=1) > 0
        
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
