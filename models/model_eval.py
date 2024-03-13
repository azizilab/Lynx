import os
import sys
import numpy as np
import torch
from scipy import sparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import nx_to_edge_index
from constants import *


def eval(model, graph, feature_mat,
               device = torch.device('cpu')):
    
    model = model.to(device)
    x = torch.tensor(feature_mat)
    x = x.float().to(device)
    edge_index = nx_to_edge_index(graph).to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    z = z.detach().cpu().numpy()
    return z
