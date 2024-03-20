import os
import sys
import numpy as np
import torch
from scipy import sparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import nx_to_edge_attrs
from constants import *


def eval(model, graph, feature_mat,
               device = torch.device('cpu')):
    
    model = model.to(device)
    x = torch.tensor(feature_mat)
    x = x.float().to(device)
    edge_index, edge_weight = nx_to_edge_attrs(graph)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_weight)
    z = z.detach().cpu().numpy()
    return z
