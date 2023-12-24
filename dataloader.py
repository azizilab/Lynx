import os
import cv2
import numpy as np
import tifffile
import gcsfs

from skimage.transform import rescale
from collections import OrderedDict
from typing import Optional, Set, List, Dict

from __init__ import LOGGER


class CyIFReader:
    """
    Load, preprocess & apply cycle-wise registration on
    CyCIF multiplexed images from gcloud
    
     - Naming format: `CyIF_{slide #}_{cycle #}_{tissue #}.qptiff`

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
        (Optional) Down-scale ratio for within-cycle registration
    """

    def __init__(
        self,
        credential_path,
        bucket_id,
        project_id,
        home_path,
        **kwargs,
    ):
        # Additional kwargs
        self.params = {
            'scale': 0.25,
            'env_key': 'GOOGLE_APPLICATION_CREDENTIALS'
        }
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v
        
        os.environ[self.params['env_key']] = credential_path
        
        self.bucket_id = bucket_id
        self.home_path = home_path
        self.gcs = gcsfs.GCSFileSystem(project=project_id,
                                       access='read_write',
                                       token=os.getenv(self.params['env_key']))
        self.chan_annots = {
            'Opal 520': {1: 'B-catenin-AF 488', 2: 'Pan CK', 3: 'CD45', 4: 'CD56'},
            'Opal 570': {1: 'GS 647', 2: 'Col I', 3: 'Arg1', 4: 'PU1'},
            'Opal 690': {1: 'ASS1 PE', 2: 'CD31', 3: 'CD68', 4: 'Vimentin'},
            'Opal 780': {1: 'CYP3A4', 3: 'Lyve1', 4: 'CD3'}
        }

        # Read slide ids
        self.slide_ids = sorted([
            path.rpartition('/')[-1]
            for path in self.gcs.ls(os.path.join(self.bucket_id, self.home_path))
            if self.gcs.isdir(path) and 'CyIF' in path
        ])

    def load_imgs(
        self, 
        slide_id,
        chans_to_ignore: Set = {},
        verbose: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load CyIF `tiff` images under the given slide ID with channel names:

            A single `slide_id` contains M tissue sections (Z-slice)
            A tissue section contains N imaging cycles / scans / rounds
            A imaging cycle contains K imaging channels

        Returns
        -------
        named_imgs : dict[str, dict[str, np.ndarray]]
            Annotated images as dictionary
            Outer key: file name for each tiff img
            Inner key: channel IDs
            Value: 2-D image pixel intensities

        """
        assert slide_id in self.slide_ids, \
            "Slide {} doesn't exist".format(slide_id)

        LOGGER.info('Loading images from Slide {}...'.format(slide_id))
        
        # Load filenames & channel annotations
        slide_path = os.path.join(self.bucket_id, self.home_path, slide_id)
        file_list = []
        for cycle in sorted(self.gcs.ls(slide_path)):
            if self.gcs.isdir(cycle) and 'Scan-0' not in cycle:  # Skip AF round for now
                file_list.extend(
                    sorted([
                        f for f in self.gcs.ls(cycle)
                        if f[-6:] == 'qptiff'
                    ])
                )
        
        chan_list = [
            self._load_chan_labels(file_path)
            for file_path in file_list
        ]

        named_imgs = {}
        for (file_path, chan_lbls) in zip(file_list, chan_list):
            filename = file_path.rpartition('/')[-1]
            cycle_id = filename.split('_')[2]
            if verbose:
                LOGGER.info('\tLoading {}...'.format(filename))

            named_img = {}
            ifile = self.gcs.open(file_path, 'rb')
            img = tifffile.imread(ifile)
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
                    named_img[chan_lbl] = np.round(255*adj_val).astype(np.uint8)

            # TODO: debugging on whether to denoise img by subtracting AF
            # named_imgs[filename] = self._denoise(named_img)
            # named_img.pop('Sample AF', None)
            named_imgs[filename] = named_img
            ifile.close()

        return named_imgs
    
    def register_cycles(
        self, 
        named_imgs: Dict[str, Dict[str, np.ndarray]],
        verbose=True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Apply affine registration via SIFT, warp channels from different
        rounds /cycles / scans to get the stacked multiplexed output

        Parameters
        ----------
        named_imgs : dict[str, dict[str, np.ndarray]]
            Dict. of annotated images 
            Outer key: file name for each tiff img
            Inner key: channel IDs
            Value: 2-D images

        Returns
        -------
        warped_imgs : dict[str, dict[str, np.ndarray]]
            Dict. of registered images to the Cycle #1
            Outer key: file name for each tissue (Z-slice)
            Inner key: annotated channel IDs
            Value: 2-D warped images
        """
        # Parse filenames within the same tissue (Z-slice)
        tiss_dict = OrderedDict()
        for filename in named_imgs:
            tid = filename.split('.')[0][-2:]
            tiss_dict.setdefault(tid, []).append(filename)

        # Registration
        slide_id = next(iter(named_imgs)).split('_')[1]
        warped_imgs = {}

        for tid, fnames in tiss_dict.items():
            if verbose:
                LOGGER.info('Registering channels from Slide {0}, Tissue {1}...'.format(
                    fnames[0].rpartition('/')[-1][:7], tid
                ))

            dapi_dst = named_imgs[fnames[0]]['DAPI']
            size = dapi_dst.shape  
            dapi_stacked = [dapi_dst]
            warped_img = {}

            # Append channels from the ref. image (Cycle #1)
            for chan_lbl, img in named_imgs[fnames[0]].items():
                if chan_lbl != 'DAPI':
                    annot = self._get_cid(chan_lbl, 1)  
                    warped_img[annot] = img

            for fname in fnames[1:]:
                named_img = named_imgs[fname]
                cycle_id = int(fname.split('_')[2])

                # Register DAPI to get affine matrix
                dapi_src = named_img['DAPI']
                H = self.get_affine_matrix(dapi_src, dapi_dst)
                dapi_warped = self.warp(dapi_src, size, H)
                dapi_stacked.append(dapi_warped)

                # Register remaining channels from Cycle #2-n
                for chan_lbl, img_src in named_img.items():
                    annot = self._get_cid(chan_lbl, cycle_id)
                    warped_img[annot] = self.warp(img_src, size, H)

            # TODO: verify whether taking `dapi_src` or use overlaid DAPI w/ MIP
            warped_img['DAPI'] = np.array(dapi_stacked).max(0)

            # (Z-slice count): (# tissue per slide) * (slide_id) + (tissue_id)
            z = len(tiss_dict) * (int(slide_id)-1) + int(tid)
            slc_key = 'CyIF_tiss_' + (str(z) if z >= 10 else '0'+str(z))
            warped_imgs[slc_key] = warped_img

        return warped_imgs
    
    @staticmethod
    def save_annotated_tiff(annot_imgs, output_path, verbose=True):
        """
        Save the multi-channel image as an annotated OME-TIFF file
        """
        # TODO: save multiplexed images directly to gcloud
        if os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        for tid, annot_img in annot_imgs.items():
            channel_names = list(annot_img.keys())
            channel_intensities = list(annot_img.values())
            img = np.array(channel_intensities)
            filename = tid + '.ome.tiff'

            if verbose:
                LOGGER.info('Saving {0}-chan image {1}...'.format(img.shape[0], filename))

            tifffile.imwrite(
                os.path.join(output_path, filename), 
                img, 
                metadata={
                    'axes': 'CYX', 
                    'Channel': {'Name': channel_names}
                }
            )
        return None
    
    @staticmethod
    def get_affine_matrix(
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Compute 3x3 Affine transformation matrix by registering 
        `source` image against `target` (SIFT)
        """
        img_src = source.copy()
        img_dst = target.copy()

        if img_src.max() == 1:
            img_src = np.round(img_src*255).astype(np.uint8)
        if img_dst.max() == 1:
            img_dst = np.round(img_dst*255).astype(np.uint8)

        sift = cv2.SIFT_create()
        pts_src, des_src = sift.detectAndCompute(img_src, None)
        pts_dst, des_dst = sift.detectAndCompute(img_dst, None)
        
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des_src, des_dst, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        
        pts1 = np.float32([pts_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([pts_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        return H
    
    @staticmethod
    def warp(
        img_src: np.ndarray,
        dst_shape,
        H: np.ndarray = np.identity(3)
    ):
        return cv2.warpPerspective(img_src, H, (dst_shape[1], dst_shape[0]))
    
    def _get_cid(self, lbl, cycle_id):
        """Get annotated channel id"""
        return self.chan_annots[lbl][cycle_id] \
               if  lbl in self.chan_annots \
               else lbl
    
    def _denoise(self, named_img):
        """Subtract AF channel"""
        for chan_lbl, val in named_img.items():
            named_img[chan_lbl] = np.max(
                [val - named_img['Sample AF'], np.zeros_like(val)],
                axis=0
            )
        named_img.pop('Sample AF', None)
        return named_img

    def _load_chan_labels(self, filename):
        labels = None
        ifile = self.gcs.open(filename, 'rb')
        tif = tifffile.TiffFile(ifile)

        try:
            metadata = tif.pages[0].tags.get('IJMetadata').value
            labels = metadata['Labels']
        except (AttributeError, KeyError):
            print('Error retrieving metadata from {}'.format(filename))
        ifile.close()
        tif.close()
        
        return labels
    