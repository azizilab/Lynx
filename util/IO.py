import os
import sys
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import gcsfs
import gzip
import xml.etree.ElementTree as ET

from skimage.transform import rescale
from skimage.filters import gaussian as gaussian_blur
from skimage.morphology import dilation
from collections import OrderedDict
from typing import Optional, Set, List, Dict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from __init__ import LOGGER
from registration import get_affine_matrix, affine_warp
from utils import get_roi_mask, norm_by_channel, get_PCs


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
    """
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
    path: str, 
    raw_count: bool = True, 
    min_counts: int = 10, 
    min_cells: int = 5
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
        
        adata.obs = meta_df.copy()
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(1).A.flatten()
        
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        adata.obs['library_size'] = adata.X.A.sum(1)
    
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()  # XY-index
    load_spatial_metadata(adata, path=os.path.join(path, 'morphology_focus.ome.tif'), load_img=True)  # Append hi-res image

    return adata


def load_desi(
    filename, 
    raw_img=True,
    dilate=True
):
    if raw_img:
        assert os.path.exists(filename), \
            "DESI path {} doesn't exist".format(filename)

        img = norm_by_channel(tifffile.imread(filename))
        if dilate:
            img = [dilation(chan, footprint=np.ones((3, 3))) for chan in img]

        # Load image
        roi_mask = get_roi_mask(img)
        adata = sc.AnnData(img[:, roi_mask].T)
        load_spatial_metadata(adata, load_img=False)
        
        coords = np.asarray(np.nonzero(roi_mask)) # YX-index, dim: [2, Y*X]
        adata.obs['y_centroid'], adata.obs['x_centroid'] = coords
        adata.obsm['spatial'] = np.array([coords[1], coords[0]]).T  # XY-index
        adata.obsm['X_img'] = img

        # Load feature annotations
        try:
            mz_labels = load_ome_labels(filename)
            adata.var_names = mz_labels
        except ET.ParseError:
            pass

    else:
        # Load preprocessed adata
        if filename[-4:] != 'h5ad':
            filename += '.h5ad'
        assert os.path.exists(filename), \
            "DESI path {} doesn't exist".format(filename)
        adata = sc.read_h5ad(filename)
    
    load_spatial_metadata(adata, load_img=False)  # Load dummy `uns['spatial']`
    return adata


def load_ab_stain(filename, adata_ref):
    """
    Load multiplexed antibody staining image as `sc.AnnData`
    """
    # Load raw images, skip DAPI channel
    img = tifffile.imread(filename)[1:]

    # Preprocessing individual channels
    for i, chan in enumerate(img):
        img[i] = gaussian_blur(
            dilation(
                chan, footprint=np.ones((3, 3))
            ),
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
    metric: str ='barcode',  
    ratio: float = 1.0         
):
    """
    Filter common cells across 2 spatial modalities

    Parameters
    ----------
    adata_ref : sc.AnnData
        Expression of the hi-res `reference` modality (e.g. Xenium)
    adata_src : sc.AnnData
        Expression of the low-res `source` modality (e.g. DESI)
    metric : str
        Filtering criteria (by `barcode` or `coord`)
    ratio : float
        Coordinate mapping ratio (ref --> src)

    Both adata objects contains mapped coordinates
    [x_centroids, y_centroids] under `adata.obsm`

    Returns
    -------
    (adata_ref_filtered, adata_src_filtered)
    """
    assert metric == 'barcode' or metric == 'coord', "Filtering criteria: `barcode` or `coord`"

    if metric == 'barcode':
        barcodes = np.intersect1d(adata_ref.obs_names, adata_src.obs_names)
        assert len(barcodes) > 0, "0 common cell barcode found, try filtering by coordinates"

        return adata_ref[barcodes, :], adata_src[barcodes, :]

    else:
        assert 'X_img' in adata_src.obsm_keys(), "Source image required to filter by coordinates"
        src_img = adata_src.obsm['X_img']
        coords = np.round(
            adata_ref.obs[['y_centroid', 'x_centroid']].copy().to_numpy().T * ratio
        ).astype(np.int16)  # dim: [2, Y*X], YX-index

        adata_src_filtered =  sc.AnnData(
            np.array([chan[tuple(coords)] for chan in src_img]).T
        )
        adata_src_filtered.obs['x_centroid'] = coords[1]
        adata_src_filtered.obs['y_centroid'] = coords[0]
        adata_src_filtered.obsm['spatial'] = adata_src_filtered .obs[['x_centroid', 'y_centroid']].values
        adata_src_filtered.var_names = adata_src_filtered .var_names

        load_spatial_metadata(adata_src_filtered, load_img=False)
        return adata_ref, adata_src_filtered
    

def load_spatial_metadata(adata, path=None, load_img=True):
    """
    Append the corresponding spatial image to ISS/ISH expression matrix
    
    Parameters
    ----------
    adata : sc.AnnData
        ISS/ISH expression matrix (e.g. Xenium, MERFISH)
    scale : float
        Downscale ratio for hi-res image
    """
    if path:
        assert os.path.isfile(path), "Unable to find corresponding image\n {}".format(path)
        sample_id = path.strip('/').split('/')[-2] if len(path.strip('/').split('/')) > 2 else 'sample'
        img = tifffile.imread(path) if load_img else None
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
    else:
        sample_id = 'sample'
        img = None  # Placeholder w/ empty entry for `adata.uns`
        
    adata.uns['spatial'] = {sample_id: {'images': {'hires': img}, 
                                        'scalefactors': {'spot_diameter_fullres': 1.0, 
                                                         'tissue_hires_scalef': 1.0}}}
    return None


def load_multiomics(
    sample_id: str,
    ref_path: str,
    src_path: str,
    mdata_df: pd.DataFrame = None,
    n_pcs: int = 10,
    verbose: bool = True
):
    """
    Load and preprocess expressions of paired spatial multi-omics data

    Parameters
    ----------
    sample_id : str
        shared `sample_id` across the multi-omics data
    ref_path : str
        hi-res `reference` modality (e.g. Xenium)
    src_path : str
        low-res `source` modality (e.g. DESI)
    n_pcs : int
        # PCs for source modality dim. reduction
    mdata_df : pd.DataFrame
        Optional sample-specific covariate info.
    """
    if verbose:
        LOGGER.info("Loading paired samples of {}...".format(sample_id))

    adata = load_xenium(os.path.join(ref_path, sample_id))
    adata_desi = load_desi(os.path.join(src_path, sample_id), raw_img=False)
    adata, adata_desi = filter_cells(adata, adata_desi)

    # Load aux. variable (u) & covariate design matrix (s)
    get_PCs(adata_desi, n_pcs=n_pcs)
    adata.obsm['X_aux'] = adata_desi.obsm['X_pca'].astype(np.float32)
    if mdata_df is not None:
        adata.obsm['X_s'] = np.tile(
            mdata_df.loc[sample_id].to_numpy(),
            (adata.shape[0], 1)
        )
    return adata


def save_annot_tiffs(annot_imgs, path, verbose=True):
    """
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
    

class GcloudReader:
    """
    Load tif images from gcloud

    Parameters
    ----------
    credential_path : str
        JSON credential file for gcloud connection
    bucket_id : str
        gcloud bucket ID
    project_id : str
        gcloud project ID
    home_path : str
        home directory for data fetching
    scale : float
        (Optional) Down-scale ratio
    """
    def __init__(
        self,
        credential_path,
        project_id,
        bucket_id,
        home_path,
        **kwargs,
    ):
        # Additional kwargs
        self.params = {
            'scale': 1,                                                                              
            'env_key': 'GOOGLE_APPLICATION_CREDENTIALS'
        }

        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v
        
        os.environ[self.params['env_key']] = credential_path
        
        self.bucket_id = bucket_id
        self.home_path = os.path.join(bucket_id, home_path)
        self.gcs = gcsfs.GCSFileSystem(project=project_id,
                                       access='read_write',
                                       token=os.getenv(self.params['env_key']))
        
    def load_imgs(
        self, 
        path: str = None, 
        ext: str = 'tif'
    ):
        """
        Load CyIF `tiff` images under the given directory

        Returns
        -------
        (Optional) imgs : list[np.ndarray]
            List of Tifffile images
        (Optional) annot_imgs : dict[str, dict[str, np.ndarray]]
            Annotated images as dictionary
            Outer key: file name for each tiff img
            Inner key: channel IDs
            Value: 2-D image pixel intensities
        """
        assert 'tif' in ext, "Support `tif` I/O only"
        is_annotated = False 
        path = self.home_path if path is None else \
            os.path.join(self.home_path, path)
        
        filenames = [f for f in sorted(self.gcs.ls(path))
                     if f[-len(ext):] == ext]
        
        # Load images
        imgs = []
        for file_path in filenames:
            img = tifffile.imread(self.gcs.open(file_path, 'rb'))
            if self.params['scale'] != 1:
                img = rescale(img,
                              scale=self.params['scale'],
                              preserve_range=True,
                              channel_axis=0)
                
            # Rescale each channel's intensity to [0-255]
            for i, chan in enumerate(img):
                if chan.min() != 0 or chan.max() != 255:
                    adj_val = (chan-chan.min()) / (chan.max()-chan.min())
                    img[i] = np.round(255*adj_val).astype(np.uint8)

            imgs.append(img)

        # Load annotations
        annot_imgs = {}
        if 'qptiff' in ext or 'ome.tif' in ext:
            is_annotated = True
            chan_list = [self._load_chan_labels(file_path)
                         for file_path in filenames]  
            for (file_path, chan_lbls, img) in zip(filenames, chan_list, imgs):
                filename = file_path.rpartition('/')[-1]
                annot_img = {chan_lbl: chan
                             for (chan_lbl, chan) in zip(chan_lbls, img)}
                annot_imgs[filename] = annot_img

        return annot_imgs if is_annotated else imgs
        
    def _load_chan_labels(self, filename):
        assert 'qptiff' in filename or 'ome.tif' \
            "Annotation format should be QPTIFF or OME-TIFF"
        try:
            ifile = self.gcs.open(filename, 'rb')
            return load_qp_labels(filename) if 'qptiff' in filename else \
                   load_ome_labels(filename)
        except FileNotFoundError:
            print("{} doesn't exist".format(filename))
        return None
        
    @staticmethod
    def save_annotated_imgs(annot_imgs, output_path, verbose=True):
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
