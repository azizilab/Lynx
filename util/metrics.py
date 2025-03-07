# Metrics for spatial trajectory / clustering inference

import numpy as np
from typing import Iterable
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score


def compute_auroc(
    gamma : np.ndarray,
    antibodies : np.ndarray,
    signs: Iterable[str] = ['-', '-', '+', '+']
):
    r"""Compute average precision score between spatial trajectory (\gamma) 
    predictions and zonation-specific antibody channels as various thresholds
     - signs: binarizing sign corresponding to specific antibody channel ('-': <=, '+': >)
     - antibody_thresholds ([0.1,..., 0.9]): binarizing each antibody channel
    """
    assert len(gamma) == len(antibodies), \
        "Inconsistent # data points btw spatial trajectory & antibody image"

    antibody_thresholds = np.linspace(0.1, 0.9, 9)
    n_thresholds = len(antibody_thresholds)
    n_channels = antibodies.shape[-1]
    aucs = np.zeros((n_thresholds, n_channels))

    for i, antibody_thld in enumerate(antibody_thresholds):
        for j, chan in enumerate(antibodies.T):
            sign = signs[j]
            y_true = (chan > antibody_thld)
            aucs[i, j] = roc_auc_score(y_true, gamma) if sign == '+' \
                else roc_auc_score(y_true, 1.-gamma)
                
    return aucs

