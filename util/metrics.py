# Metrics for spatial trajectory / clustering inference
import os 
import sys
import numpy as np
from typing import Iterable
from sklearn.metrics import average_precision_score, roc_auc_score
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

