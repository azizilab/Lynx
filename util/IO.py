import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import gzip
import xml.etree.ElementTree as ET

from skimage.transform import rescale
from skimage.filters import gaussian as gaussian_blur
from skimage.morphology import dilation
from collections import OrderedDict
from typing import Optional, Set, List, Dict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from __init__ import LOGGER


# -------------------
#  IO util functions
# -------------------

def load_qp_labels(filename):
    try:
        ifile = open(filename, 'rb')
        tif = tifffile.TiffFile(ifile)
        metadata = tif.pages[0].tags.get('IJMetadata').value
        labels = metadata['Labels']

        ifile.close() 
        tif.close()
        return [l.strip() for l in labels]

    except (AttributeError, KeyError):
        print('Error retrieving metadata from {}'.format(filename))
        
    return None


def load_ome_labels(filename):
    try:
        ifile = open(filename, 'rb')
        tif = tifffile.TiffFile(ifile)
        desc = ET.fromstring(tif.pages[0].tags['ImageDescription'].value)
        tree = ET.ElementTree(desc)

        # Check XML tag name for `Channel`
        labels = [elem.get('Name')
                  for elem in tree.iter()
                  if 'Channel' in elem.tag]
        # labels = [s + f' (C{i+1})' for i, s in enumerate(labels)]
        
        ifile.close()
        tif.close()
        return [l.strip() for l in labels]

    except (AttributeError, KeyError):
        print('Error retrieving metadata from {}'.format(filename))

    return None


def load_annot_tiffs(file_path, ext='ome.tif'):
    r"""
    Load annotated Tiff images from directory

    Returns
    -------
    annot_imgs : dict[str, dict[str, np.ndarray]]
        Annotated images as dictionary
        Outer key: file name for each tiff img
        Inner key: channel IDs
        Value: 2-D image pixel intensities
    """
    assert ext == 'qptiff' or 'ome.tif' in ext, \
        "Extension should be QPTIFF / OME-TIFF format"
    filenames = [f for f in sorted(os.listdir(file_path))
                 if f[-len(ext):] == ext]
    
    annot_imgs = {}
    for f in filenames:
        path = os.path.join(file_path, f)
        img = tifffile.imread(path)
        labels = load_qp_labels(path) if ext == 'qptiff' else \
                 load_ome_labels(path)
        
        annot_imgs[f] = {lbl: chan 
                         for (lbl, chan) in zip(labels, img)}
    return annot_imgs


def check_ion_mode(metabolite, pos_path, neg_path):
    r"""Check whether the given metabolite ion is from +/- mode"""
    metabolite = metabolite.strip()

    # Load all +/- ion labels from sample file
    pos_labels = load_ome_labels(pos_path)
    pos_ions = [ion if 'm/z' in ion else ion.strip() for ion in pos_labels]
    neg_labels = load_ome_labels(neg_path)
    neg_ions = [ion if 'm/z' in ion else ion.strip() for ion in neg_labels]

    if metabolite in pos_ions: 
        return '+'
    elif metabolite in neg_ions:
        return '-'
    else:
        print('Unannotated ion:', metabolite)
        return 'NA'


def load_anchor_points(path):
    r"""Load anchor points for Affine Transformation"""
    assert os.path.exists(path),\
        "Directory {} doesn't exist".format(path)

    filenames = [f for f in sorted(os.listdir(path))
                 if f[-3:] == 'pts']
    points = []
    for filename in filenames:
        pts = np.loadtxt(os.path.join(path, filename))
        points.append([tuple(pt) for pt in pts])
    return points


def load_xenium(
    path : str,
    filename : Optional[str] = None,
    raw_count : bool = True, 
    min_counts : int = 20, 
    min_cells : int = 5,
    load_metadata : bool = True,
    load_img : bool = False
):
    if filename is None:
        filename = 'cell_feature_matrix.h5' if raw_count else 'filtered_feature_matrix.h5'
    elif 'filtered' in filename:
        print('Loading filtered feature matrix:', filename)
    assert os.path.exists(path), \
        "Xenium path {} doesn't exist".format(path)
    assert os.path.isfile(os.path.join(path, filename)), \
        """Feature matrix {} doesn't exist\n,
           Please set `raw_count=False` if the filtered / normalized 
           feature matrix are saved under the same directory""".format(filename)

    # Load AnnData
    try:
        adata = sc.read_10x_h5(os.path.join(path, filename))
    except ValueError:
        adata = sc.read_h5ad(os.path.join(path, filename))   # legacy / custom .h5 file

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)          

    # Load spatial metadata if not appended to `.h5` file yet
    if load_metadata and 'spatial' not in adata.obsm_keys():
        with gzip.open(os.path.join(path, 'cells.csv.gz'), 'rt') as ifile:
            meta_df = pd.read_csv(ifile, index_col=[0])
            meta_df.index = meta_df.index.astype(str) 
        adata.obs = pd.concat([adata.obs, meta_df.loc[adata.obs_names]], axis=1, join='outer')
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()  # XY-index
    
    img_filename = ''
    if load_img:
        if os.path.isfile(os.path.join(path, 'morphology_mip.ome.tif')):
            img_filename = 'morphology_mip.ome.tif'
        elif os.path.isfile(os.path.join(path, 'morphology_focus.ome.tif')):
            img_filename = 'morphology_focus.ome.tif'
        elif os.path.isfile(os.path.join(path, 'morphology.ome.tif')):
            img_filename = 'morphology.ome.tif'
        else:
            raise FileNotFoundError(
                'Please ensure the Xenium directory contains the associated fluorescent image',
                ' - morphology_mip.ome.tif',
                ' - morphology_focus.ome.tif',
                ' - morphology.ome.tif'
            )
    
    load_spatial_metadata(
        adata, 
        path=os.path.join(path, img_filename), 
        load_img=load_img
    )
    
    return adata


def load_ab_stain(filename, adata_ref):
    r"""
    Load multiplexed antibody staining image as `sc.AnnData`
    """
    # Load raw images, skip DAPI channel
    img = tifffile.imread(filename)[1:]

    # Preprocessing individual channels
    for i, chan in enumerate(img):
        img[i] = gaussian_blur(
            dilation(chan, footprint=np.ones((3, 3))),
            sigma=5
        )

    # Filter indices mapped to reference `adata`
    coords = np.round(
        adata_ref.obs[['y_centroid', 'x_centroid']].copy().to_numpy().T
    ).astype(np.int16)  # dim: [Y*X, 2], YX-index 

    adata = sc.AnnData(
        np.array([chan[tuple(coords)] for chan in img]).T
    )
    adata.obs['y_centroid'], adata.obs['x_centroid'] = coords
    adata.obsm['spatial'] = np.array([coords[1], coords[0]]).T  # XY-index
    adata.obs.index = adata_ref.obs.index.copy()
    load_spatial_metadata(adata, load_img=False)

    try:
        labels = load_ome_labels(filename)[1:]
        adata.var_names = labels
    except ET.ParseError:
        pass

    return adata


def filter_cells(
    adata_ref: sc.AnnData, 
    adata_query: sc.AnnData,
    by: str ='map',  
    filter_ref: bool=True,
):
    r"""
    Filter common cells across 2 spatial modalities

    Parameters
    ----------
    adata_ref : sc.AnnData
        Expression matrix of `ref` modality
    adata_query : sc.AnnData
        Expression matrix of `query` modality
    by : str
        Filtering option (by `barcode` / `map`)
    

    Both adata objects contains mapped coordinates
    [x_centroids, y_centroids] under `adata.obsm`

    Returns
    -------
    (adata_ref_filtered, adata_query_filtered)
    """
    assert by == 'barcode' or by == 'coord' or by == 'map', \
        "Filtering criteria: `barcode` or `map`"

    if by == 'barcode':
        barcodes = np.intersect1d(adata_ref.obs_names, adata_query.obs_names)
        assert len(barcodes) > 0, "0 common cell barcode found, try filtering by coordinates"
        ref_indices = barcodes
        query_indices = barcodes
    
    elif by == 'map':
        # Filter by taking <==> intersect of pre-computed 
        # cell (ref) - pixel (query) mapping
        ref_map = set()
        ref_indices = []
        for i, coord in enumerate(adata_ref.obsm['desi_map']):
            if not np.array_equal(coord, [-1, -1]):
                ref_map.add(tuple(coord))
                ref_indices.append(i)
        query_indices = [
            i for i, coord in enumerate(adata_query.obsm['spatial'])
            if tuple(coord) in ref_map
        ]

    else:
        raise NotImplementedError(by)

    adata_ref_filtered = adata_ref[ref_indices].copy() if filter_ref else adata_ref.copy()
    adata_query_filtered = adata_query[query_indices].copy()

    # Reset the filtered `.obs_names`
    # adata_ref_filtered.obs_names = adata_ref.obs_names[ref_indices]  # barcode
    adata_query_filtered.obs.reset_index(drop=True, inplace=True)
    adata_query_filtered.obs_names = adata_query_filtered.obs_names.astype(str)  # 0-index
    
    return adata_ref_filtered, adata_query_filtered
    

def load_spatial_metadata(adata, path='', load_img=False):
    r"""
    Append the corresponding spatial image to ISS/ISH expression matrix
    """
    if os.path.isfile(path) and load_img:
        sample_id = path.strip('/').split('/')[-2] if len(path.strip('/').split('/')) > 2 else 'sample'
        img = tifffile.imread(path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
    else:
        sample_id = 'sample'
        img = None  # Placeholder w/ empty entry for `adata.uns`
        
    adata.uns['spatial'] = {
        sample_id: {
            'images': {'hires': img}, 
            'scalefactors': {
                'spot_diameter_fullres': 1.0, 
                'tissue_hires_scalef': 1.0
            }
        }
    }
    return None

  
def save_annot_tif(file, img, annots):
    r"""
    Save individual annotated image (dim: [C, Y, X])
    """
    assert img.ndim == 3 and img.shape[0] == len(annots), \
        "Image dim != [C, Y, X] or image channel != annotation length"
    
    path = file.rpartition('/')[0]
    if os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if 'ome.tif' not in file:
        file += '.ome.tif'
    
    tifffile.imwrite(
        file,
        img,
        metadata={
            'axes': 'CYX',
            'Channel': {'Name': annots}
        }
    )


def save_annot_tifs(annot_imgs, path, verbose=True):
    r"""
    Save a list of multi-channel images as annotated OME-TIFF files

    Parameters
    -------
    annot_imgs : dict[str, dict[str, np.ndarray]]
        Annotated images as dictionary
        Outer key: file name for each tiff img
        Inner key: channel IDs
        Value: 2-D image pixel intensities

    path : str
        Output directory
    """
    if os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for tid, annot_img in annot_imgs.items():
        channel_names = list(annot_img.keys())
        channel_intensities = list(annot_img.values())
        img = np.array(channel_intensities)

        if 'ome.tif' not in tid:
            tid += '.ome.tif'

        if verbose:
            LOGGER.info('Saving {0}-chan image {1}...'.format(img.shape[0], tid))

        tifffile.imwrite(
            os.path.join(path, tid), 
            img, 
            metadata={
                'axes': 'CYX', 
                'Channel': {'Name': channel_names}
            }
        )
    return None   
