#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np 
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

def load_qp_labels(ifile, filename):
    try:
        tif = tifffile.TiffFile(ifile)
        metadata = tif.pages[0].tags.get('IJMetadata').value
        labels = metadata['Labels']

        ifile.close() 
        tif.close()
        return labels

    except (AttributeError, KeyError):
        print('Error retrieving metadata from {}'.format(filename))
        
    return None

def load_ome_labels(ifile, filename):
    try:
        tif = tifffile.TiffFile(ifile)
        desc = ET.fromstring(tif.pages[0].tags['ImageDescription'].value)
        tree = ET.ElementTree(desc)

        # Check XML tag name for `Channel`
        labels = [elem.get('Name')
                  for elem in tree.iter()
                  if 'Channel' in elem.tag]
        
        ifile.close()
        tif.close()
        return labels

    except (AttributeError, KeyError):
        print('Error retrieving metadata from {}'.format(filename))

    return None

def load_annot_tiffs(file_path, ext='ome.tif'):
    assert ext == 'qptiff' or 'ome.tif' in ext,         "Extension should be QPTIFF / OME-TIFF format"
    filenames = [f for f in sorted(os.listdir(file_path))
                 if f[-len(ext):] == ext]
    annot_imgs = []
    for f in filenames:
        img = tifffile.imread(os.path.join(file_path, f))
        ifile = open(os.path.join(file_path, f), 'rb')
        labels = load_qp_labels(ifile, f) if ext == 'qptiff' else                  load_ome_labels(ifile, f)
        annot_imgs.append({lbl: chan 
                           for (lbl, chan) in zip(labels, img)})
    return annot_imgs, filenames
    
def save_annotated_tiff(annot_imgs, output_path, verbose=True):
    """
    Save the multi-channel image as an annotated OME-TIFF file
    """
    if os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for tid, annot_img in annot_imgs.items():
        channel_names = list(annot_img.keys())
        channel_intensities = list(annot_img.values())
        img = np.array(channel_intensities)

        if 'ome.tif' not in tid:
            tid += '.ome.tif'

        if verbose:
            LOGGER.info('Saving {0}-chan image {1}...'.format(img.shape[0], tid))

        tifffile.imwrite(
            os.path.join(output_path, tid), 
            img, 
            metadata={
                'axes': 'CYX', 
                'Channel': {'Name': channel_names}
            }
        )
    return None
        
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
    af_data = pd.read_csv('autof.csv')
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
        save_annotated_tiff(dict_processed, output_path, verbose=False)

def manual_correction(input_dir, output_path):
    """This function corrects for AF using manual thresholding set at adjustable percentile:
    input: input directory and output directory paths.
    output: ome.tif AF corrected images with same file names in the output directory.
    Should be applied only to images that have very high intensity spots that cannot be removed from otsu."""
    
    annot_imgs, filenames = load_annot_tiffs(input_dir, ext='ome.tif')
    af_data = pd.read_csv('autof.csv')
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
        save_annotated_tiff(dict_processed, output_path, verbose=False)

def bcatenin_correction(input_dir, output_path):
    """This function corrects for beta-catenine channel bleeding into the subsequent ASS1 channel:
    input: input directory and output directory paths.
    output: ome.tif AF corrected images with same file names in the output directory.
    Should be applied only to images with gradient issues"""
    
    annot_imgs, filenames = load_annot_tiffs(input_dir, ext='ome.tif')
    af_data = pd.read_csv('autof.csv')
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
        save_annotated_tiff(dict_processed, output_path, verbose=False)

