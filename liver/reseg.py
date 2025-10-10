#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pipeline for Cellpose Multimodal re-segmentation on Xenium

import os
import gc
import torch
import tifffile
import numpy as np

from scipy import ndimage as ndi
from cellpose import models
from skimage.measure import label as skimage_label
from skimage.morphology import remove_small_objects
from skimage.transform import resize
import argparse


# --------------------
#   Helper functions
# --------------------


# %%
def load_ab_image(data_path, sample_id, section_id):
    """Load multi-channel antibody image from Xenium"""
    morph_path = os.path.join(data_path, sample_id, section_id, 'morphology_focus')
    filenames = sorted(os.listdir(morph_path))
    
    chans = [tifffile.imread(os.path.join(morph_path, f))[i] for i, f in enumerate(filenames)]
    img = np.stack(chans, axis=0).astype(np.float32)
    
    for i in range(img.shape[0]):
        img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())

    return img


def cellpose_segment(img, factor=0.2, diam=50):
    """Multi-modal segmentation with Cellpose"""
    assert img.ndim == 3, "Input image must be multi-channel (C, Y, X)"
    nuclei_chan = img[0] + img[2]
    nuclei_chan[nuclei_chan > 255] = 255
    
    combined_img = (np.stack(
        (ndi.zoom(img[1], factor), 
         ndi.zoom(nuclei_chan, factor)),
        axis=-1
    ) * 255).astype(np.uint8)
    
    model = models.CellposeModel(gpu=True)
    masks, _, _ = model.eval(
        combined_img,
        diameter=diam*factor, 
        batch_size=4,
        flow_threshold=3.0
    )

    # Resize back to full resolution
    fullres_masks = resize(masks, img.shape[1:], order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
    return fullres_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cellpose Multimodal re-segmentation on Xenium')
    parser.add_argument('--data-path', dest='data_path', help='Path to data directory')
    parser.add_argument('--sample-id', dest='sample_id', help='Sample ID')
    parser.add_argument('--section-id', dest='section_id', help='Section ID')
    parser.add_argument('--factor', type=float, default=0.2, help='Scaling factor (default: 0.2)')
    parser.add_argument('--diam', type=int, default=50, help='Cell diameter (default: 50)')
    
    args = parser.parse_args()
    data_path = args.data_path
    sample_id = args.sample_id
    section_id = args.section_id
    factor = args.factor
    diam = args.diam

    # (1). Load multi-modal image
    img = load_ab_image(data_path, sample_id, section_id)
    gc.collect()

    # (2). Multi-modal segmentation inference:
    # - Cellpose w/ Membrane & (18S + DAPI)
    # - Binary thresholding w/ aSMA/Vimentin
    # - Append non-overlapping aSMA/Vimentin masks to Cellpose results
    cyto_masks = cellpose_segment(img, factor=factor, diam=diam)
    vim_masks = skimage_label(img[3] >= np.quantile(img[3], .999), connectivity=1)
    gc.collect()
    torch.cuda.empty_cache()

    vim_exclusive_masks = np.logical_and(vim_masks > 0, cyto_masks == 0) * vim_masks
    vim_exclusive_masks = remove_small_objects(vim_exclusive_masks, min_size=10)
    vim_exclusive_masks = skimage_label(vim_exclusive_masks, connectivity=1)   # Relabel to consecutive integers
    vim_exclusive_masks[vim_exclusive_masks > 0] += cyto_masks.max()   # Offset to avoid ID collision

    combined_masks = cyto_masks + vim_exclusive_masks

    print(f"# Cells segmented by Cellpose: {cyto_masks.max()-1}")
    print(f"# Cells segmented by aSMA/Vimentin: {vim_masks.max()-1}")
    print(f"# Cells segmented after fusion: {combined_masks.max()-1}")
 
    # (3). Save re-segmentation results
    np.save(os.path.join(data_path, sample_id, section_id, 'reseg_masks.npy'), combined_masks.astype(np.uint32))

