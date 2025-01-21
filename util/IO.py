import os
import sys
import cv2
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
from utils import get_roi_mask, norm_by_channel
from utils import get_highly_variable_metabolites


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
        return labels

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
        return labels

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


def load_anchor_points(path):
    """Load anchor points for Affine Transformation"""
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
    path, 
    raw_count=True, 
    min_counts=20, 
    min_cells=5,
    load_img=False
):
    filename = 'cell_feature_matrix.h5' if raw_count else 'filtered_feature_matrix.h5'
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

    if raw_count:
        with gzip.open(os.path.join(os.path.join(path, 'cells.csv.gz')), 'rt') as ifile:
            meta_df = pd.read_csv(ifile, index_col=[0])
        
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_genes(adata, min_cells=min_cells)

        adata.obs = meta_df.loc[adata.obs_names].copy()
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(1).A.flatten()
        adata.obs['library_size'] = adata.X.A.sum(1)
    
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()  # XY-index
    load_spatial_metadata(
        adata, 
        path=os.path.join(path, 'morphology_mip.ome.tif'), 
        load_img=load_img
    )
    
    return adata


def load_desi(
    filename, 
    raw_img=True,
    sigma=5, 
    erode_pixel=5, 
    min_area=500,
    load_img=False
):
    if raw_img and 'tif' not in filename:
        filename += '.ome.tif'
    if not raw_img and 'h5' not in filename:
        filename += '.h5ad'
    assert os.path.exists(filename), \
         "DESI path {} doesn't exist".format(filename)

    if raw_img:
        img = norm_by_channel(tifffile.imread(filename))  # dim: [C, Y, X]

        # Load raw image, filter out background & tissue border outliers
        roi_mask = get_roi_mask(img, sigma=sigma, erode_pixel=erode_pixel, min_area=min_area)
        adata = sc.AnnData(img[:, roi_mask].T)
        load_spatial_metadata(adata, load_img=load_img)
        adata.uns['X_img'] = np.einsum('cyx, yx -> cyx', img, roi_mask)
        
        coords = np.asarray(np.nonzero(roi_mask))  # YX-index, dim: [2, Y*X]
        adata.obs['x_centroid'], adata.obs['y_centroid'] = coords[1], coords[0]
        adata.obsm['spatial'] = np.array([coords[1], coords[0]]).T  # XY-index

        # Load feature annotations
        try:
            mz_labels = load_ome_labels(filename)
            adata.var_names = mz_labels
        except ET.ParseError:
            pass

    else:
        # Load preprocessed adata
        adata = sc.read_h5ad(filename)
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()

    # TODO: add highly-variable feature filtering


    # Load dummy `uns['spatial']`
    load_spatial_metadata(adata, load_img=load_img)  
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
    load_spatial_metadata(adata, load_img=False)

    try:
        labels = load_ome_labels(filename)[1:]
        adata.var_names = labels
    except ET.ParseError:
        pass

    return adata


def filter_cells(
    adata_ref: sc.AnnData, 
    adata_src: sc.AnnData,
    by: str ='barcode',  
    ratio: float = 1.0         
):
    r"""
    Filter common cells across 2 spatial modalities

    Parameters
    ----------
    adata_ref : sc.AnnData
        Expression matrix of `ref` modality
    adata_src : sc.AnnData
        Expression matrix of `source` modality
    option : str
        Filtering option (by `barcode` / `map`)
    ratio : float
        Coordinate mapping ratio (ref --> src)

    Both adata objects contains mapped coordinates
    [x_centroids, y_centroids] under `adata.obsm`

    Returns
    -------
    (adata_ref_filtered, adata_src_filtered)
    """
    assert by == 'barcode' or by == 'coord' or by == 'map', \
        "Filtering criteria: `barcode` or `map`"

    if by == 'barcode':
        barcodes = np.intersect1d(adata_ref.obs_names, adata_src.obs_names)
        assert len(barcodes) > 0, "0 common cell barcode found, try filtering by coordinates"
        return adata_ref[barcodes, :], adata_src[barcodes, :]
    
    elif by == 'map':
        # Filter by taking <==> intersect of pre-computed 
        # cell (ref) - pixel (src) mapping
        ref_map = set()
        ref_indices = []
        for i, coord in enumerate(adata_ref.obsm['desi_map']):
            if not np.array_equal(coord, [-1, -1]):
                ref_map.add(tuple(coord))
                ref_indices.append(i)
        src_indices = [
            i for i, coord in enumerate(adata_src.obsm['spatial'])
            if tuple(coord) in ref_map
        ]
        return adata_ref[ref_indices], adata_src[src_indices]

    else:
        raise NotImplementedError(by)
    

def load_spatial_metadata(adata, path='', load_img=False):
    r"""
    Append the corresponding spatial image to ISS/ISH expression matrix
    
    Parameters
    ----------
    adata : sc.AnnData
        ISS/ISH expression matrix (e.g. Xenium, MERFISH)
    scale : float
        Downscale ratio for hi-res image
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

  
def load_multiomics(
    sample_id: str,
    ref_path: str,
    src_path: str,
    mdata_df: pd.DataFrame = None,
    n_features: int = 100,
    project: bool = False,
    verbose: bool = True
):
    r"""
    Load and filter paired spatial multi-omics data

    Parameters
    ----------
    sample_id : str
        shared `sample_id` across the multi-omics data
    ref_path : str
        hi-res `reference` modality (e.g. Xenium)
    src_path : str
        low-res `source` modality (e.g. DESI)
    n_features : int
        # top differentially expressed features from the `source` modality
    project : bool
        Whether to project `source` modality to `target` modality
        (e.g. modalities are registered without warping to the same resolution)
    mdata_df : pd.DataFrame
        Optional sample-specific covariate info.
    """
    if verbose:
        LOGGER.info("Loading paired samples of {}...".format(sample_id))

    filter_option = 'map' if project else 'barcode'
    adata_ref = load_xenium(os.path.join(ref_path, sample_id), load_img=False)
    adata_src = load_desi(os.path.join(src_path, sample_id), raw_img=project, load_img=project)
    adata_ref, adata_src = filter_cells(adata_ref, adata_src, by=filter_option)

    if n_features is None:
        src_features = adata_src.var_names
        src_indices = np.arange(adata_src.shape[1])
    else:
        hvfs = get_highly_variable_metabolites(adata_src, n_features=n_features)
        src_features = adata_src[:, hvfs].var_names
        src_indices = [
            i for i, feature in enumerate(adata_src.var_names)
            if feature in hvfs
        ]

    # Load auxiliary variable (u)
    if project:
        # project `source` modality to coordinates of mapped `target` modality
        assert 'desi_map' in adata_ref.obsm.keys(), \
            'Pre-defined coordinate mapping required for `project` multi-omics loading option'
    
        src_img = adata_src.uns['X_img']  # dim: [C, Y, X]
        projected_coords = tuple(np.flip(adata_ref.obsm['desi_map'].T, axis=0))  # dim: [2, N], YX-index
        auxiliary_expr = np.vstack([src_img[idx][projected_coords] for idx in src_indices]).T
    else:
        # `source` & `reference` modalities are interpolated to the same dimension`
        auxiliary_expr = adata_src[:, src_features].X.copy()
    
    adata_ref.obsm['X_aux'] = auxiliary_expr
    adata_ref.uns['aux_features'] = src_features

    # Load covariate design matrix (s)
    if mdata_df is not None:
        adata_ref.obsm['X_s'] = np.tile(
            mdata_df.loc[sample_id].to_numpy(),
            (adata_ref.shape[0], 1)
        )
    return adata_ref


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
