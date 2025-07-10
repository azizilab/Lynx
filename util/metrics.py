# Metrics for spatial trajectory / clustering inference
import os 
import sys
import gc

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from typing import List, Iterable
from sklearn.metrics import average_precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import to_dense_array


def get_antibody_threshold(adata, feature):
    assert feature in adata.var_names
    arr = to_dense_array(adata[:, feature].X).squeeze()
    bin_labels = arr > np.median(arr)  # initial guess
    lda = LinearDiscriminantAnalysis()
    lda.fit(arr.reshape(-1, 1), bin_labels)
    return -lda.intercept_[0] / lda.coef_[0][0]


def compute_ap(
    gamma : np.ndarray,
    antibodies : Iterable[np.ndarray],
):
    r"""Compute Average Precision (AP) score between spatial trajectory (t) 
    predictions & thresholded zonation-specific antibody channels 
    """
    assert len(gamma) == len(antibodies[0]), \
        "Inconsistent # data points btw spatial trajectory & antibody image"
    aps = np.zeros(len(antibodies))

    for i, antibody in enumerate(antibodies):
        aps[i] = average_precision_score(antibody, gamma)
    return aps


def compute_moran_I(
    adata : sc.AnnData,
    use_rep : List[str],
    n_repeats : int = 50,
    ss_ratio : float = 0.1
): 
    r"""Compute Moran's I index to quantify spatial "smoothness" of 
    the inferred gradient with bootstrapping (for consistency)
    """
    moran_Is = np.zeros((len(use_rep), n_repeats))
    n_obs = adata.shape[0]

    for j in range(n_repeats):
        # Random subset data points
        rand_indices = np.random.choice(np.arange(n_obs), int(ss_ratio*n_obs), replace=False)
        adata_ss = adata[rand_indices]
        sq.gr.spatial_neighbors(adata_ss, coord_type='generic', delaunay=True)

        for i, key in enumerate(use_rep):
            sq.gr.spatial_autocorr(
                adata_ss, attr='obs', genes=[key],
                mode='moran', transformation=False
            )
            moran_Is[i, j] = adata_ss.uns['moranI'].loc[key, 'I']

        del adata_ss
        gc.collect()

    return moran_Is
