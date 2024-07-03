import os
import sys
import cv2
import numpy as np
import tifffile
import gcsfs
import xml.etree.ElementTree as ET

from skimage.transform import rescale
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian as gaussian_blur
from collections import OrderedDict
from typing import Optional, Set, List, Dict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from __init__ import LOGGER
from registration import get_affine_matrix, affine_warp


# -------------------
#  IO util functions
# -------------------

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
        img = tifffile.imread(os.path.join(file_path, f))
        ifile = open(os.path.join(file_path, f), 'rb')
        labels = load_qp_labels(ifile, f) if ext == 'qptiff' else \
                 load_ome_labels(ifile, f)
        
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


def load_spatial(adata, path=None, load_img=True):
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
            return load_qp_labels(ifile, filename) if 'qptiff' in filename else \
                   load_ome_labels(ifile, filename)
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


class CyIFGcloudReader(GcloudReader):
    """
    Preprocess & apply cycle-wise registration on
    CyCIF multiplexed images from gcloud
    
     - Naming format: `CyIF_{slide #}_{cycle #}_{tissue #}.qptiff`
    """
    def __init__(
        self,
    ):
        super(CyIFGcloudReader, self).__init__()

        # Additional kwargs
        self.params = {
            'env_key': 'GOOGLE_APPLICATION_CREDENTIALS',
            'scale': 1,                                                               
            'sigma': 5,                                 # Gaussian filter std.
            'n_matches': 50,                            # min # matched pts for registration (SIFT)
        }
        
        self.chan_annots = {'Opal 520': {1: 'B-catenin-AF 488', 2: 'Pan CK', 3: 'CD45', 4: 'CD56'},
                            'Opal 570': {1: 'GS 647', 2: 'Col I', 3: 'Arg1', 4: 'PU1'},
                            'Opal 690': {1: 'ASS1 PE', 2: 'CD31', 3: 'CD68', 4: 'Vimentin'},
                            'Opal 780': {1: 'CYP3A4', 3: 'Lyve1', 4: 'CD3'}}
        
        # Read slide ids
        self.slide_ids = sorted([path.rpartition('/')[-1]
                                 for path in self.gcs.ls(os.path.join(self.bucket_id, self.home_path))
                                 if self.gcs.isdir(path) and 'CyIF' in path.rpartition('/')[-1]])

    def load_imgs(
        self, 
        slide_id,
        chans_to_ignore: Set = {},
        verbose: bool = True,
        ext: str = 'ome.tif'
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load CyIF `tiff` images under the given slide ID with channel names:

            A single `slide_id` contains M tissue sections (Z-slice)
            A tissue section contains N imaging cycles / scans / rounds
            A imaging cycle contains K imaging channels

        Returns
        -------
        annot_imgs : dict[str, dict[str, np.ndarray]]
            Annotated images as dictionary
            Outer key: file name for each tiff img
            Inner key: channel IDs
            Value: 2-D image pixel intensities
        """
        assert slide_id in self.slide_ids, \
            "Slide {} doesn't exist".format(slide_id)

        LOGGER.info('Loading images from Slide {}...'.format(slide_id))
        
        # Load filenames & channel annotations             
        slide_path = os.path.join(self.home_path, slide_id)
        file_list = []
        for cycle in sorted(self.gcs.ls(slide_path)):
            if self.gcs.isdir(cycle) and 'Scan-0' not in cycle:  # Skip AF round 
                file_list.extend(sorted([f for f in self.gcs.ls(cycle)
                                         if f[-len(ext):] == ext]))

        chan_list = [self._load_chan_labels(file_path)
                     for file_path in file_list]  

        annot_imgs = {} 
        for (file_path, chan_lbls) in zip(file_list, chan_list):
            filename = file_path.rpartition('/')[-1]
            cycle_id = filename.split('_')[2]
            if verbose:
                LOGGER.info('\tLoading {}...'.format(filename))

            annot_img = {}
            img = tifffile.imread(self.gcs.open(file_path, 'rb'))
            if self.params['scale'] != 1:
                img = rescale(img,
                              scale=self.params['scale'],
                              preserve_range=True,
                              channel_axis=0)
                
            # Rescale each channel's intensity to [0-255]
            for (chan, chan_lbl) in zip(img, chan_lbls):
                if chan_lbl not in chans_to_ignore:
                    if chan_lbl == 'Sample AF':
                        chan_lbl += '_' + cycle_id
                    adj_val = (chan-chan.min()) / (chan.max()-chan.min())
                    annot_img[chan_lbl] = np.round(255*adj_val).astype(np.uint8)

            annot_imgs[filename] = annot_img

        return annot_imgs
    
    def register_cycles(
        self, 
        annot_imgs: Dict[str, Dict[str, np.ndarray]],
        verbose=True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Apply affine registration via SIFT, warp channels from different
        rounds /cycles / scans to get the stacked multiplexed output

        Parameters
        ----------
        annot_imgs : dict[str, dict[str, np.ndarray]]
            Dict. of annotated images 
            Outer key: file name for each tiff img
            Inner key: channel IDs
            Value: 2-D images

        Returns
        -------
        warped_imgs : dict[str, dict[str, np.ndarray]]
            Dict. of registered images towards Cycle #1 (Ref.)
            Outer key: file name for each tissue (Z-slice)
            Inner key: annotated channel IDs
            Value: 2-D warped images
        """
        # Parse filenames within the same tissue (Z-slice)
        tiss_dict = OrderedDict()
        for filename in annot_imgs:
            tid = filename.split('.')[0][-2:]
            tiss_dict.setdefault(tid, []).append(filename)

        # Registration
        slide_id = next(iter(annot_imgs)).split('_')[1]
        warped_imgs = {}

        for tid, fnames in tiss_dict.items():
            if verbose:
                LOGGER.info('Registering channels from Slide {0}, Tissue {1}...'.format(
                    fnames[0].rpartition('/')[-1][:7], tid))

            dapi_dst = annot_imgs[fnames[0]]['DAPI']
            size = dapi_dst.shape  
            dapi_stacked = [dapi_dst]
            warped_img = {}

            # Append channels from the ref. image (Cycle #1)
            for chan_lbl, img in annot_imgs[fnames[0]].items():
                if chan_lbl != 'DAPI':
                    annot = self._get_cid(chan_lbl, 1)  
                    warped_img[annot] = img

            for fname in fnames[1:]:
                annot_img = annot_imgs[fname]
                cycle_id = int(fname.split('_')[2])

                # Register DAPI to get 1 matrix
                dapi_src = annot_img['DAPI']
                M = get_affine_matrix(dapi_src, dapi_dst, sigma=self.params['sigma'], n_matches=self.params['n_matches'])
                dapi_warped = affine_warp(dapi_src, size, M)
                dapi_stacked.append(dapi_warped)

                # Register remaining channels from Cycle #2-N
                for chan_lbl, img_src in annot_img.items():
                    annot = self._get_cid(chan_lbl, cycle_id)
                    warped_img[annot] = affine_warp(img_src, size, M)

            # TODO: verify whether taking `dapi_src` or use overlaid DAPI w/ MIP
            warped_img['DAPI'] = np.array(dapi_stacked).max(0)

            # (Z-slice count): (# tissue per slide) * (slide_id) + (tissue_id)
            z = len(tiss_dict) * (int(slide_id)-1) + int(tid)
            slc_key = 'CyIF_tiss_' + (str(z) if z >= 10 else '0'+str(z))
            warped_imgs[slc_key] = warped_img

        return warped_imgs
        
    def _get_cid(self, lbl, cycle_id):
        """Get annotated channel id"""
        return self.chan_annots[lbl][cycle_id] \
               if  lbl in self.chan_annots \
               else lbl
    
    def _denoise(self, annot_img):
        """Subtract AF channel"""
        for chan_lbl, val in annot_img.items():
            annot_img[chan_lbl] = np.max(
                [val - annot_img['Sample AF'], np.zeros_like(val)],
                axis=0
            )
        annot_img.pop('Sample AF', None)
        return annot_img
