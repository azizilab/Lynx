import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch_sparse import SparseTensor
from torch_sparse import spmm
from torch_geometric.utils import to_dense_adj


class GPCALayer(nn.Module):
    """
    Graph-regularized PCA w/ nonliear activation
    Code reference from:
    https://arxiv.org/pdf/2006.12294
    https://github.com/LingxiaoShawn/GPCANet
    """
    def __init__(self, configs, niter=50, act='relu',
                 center=False, device=torch.device('cpu')):
        super(GPCALayer, self).__init__()
        self.configs = configs
        self.c_in = configs.c_in
        self.c_out = configs.c_latent
        self.alpha = configs.alpha
        self.niter = niter
        self.center = center

        self.weight = nn.Parameter(torch.FloatTensor(self.c_in, self.c_out)).to(device)
        self.bias = nn.Parameter(torch.FloatTensor(1, self.c_out)).to(device)
        if act == 'relu':
            self.act = F.relu
        elif act == 'sigmoid':
            self.act = F.sigmoid
        elif act == 'softplus':
            self.act = F.softplus
        else:
            raise NotImplementedError("Unimplemented activation function {}".format(act))

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, data):
        # DEBUG: allow learnable `W``
        edge_index, x = data.edge_index, data.x
        n = x.shape[0]
        A = self._get_sparse_adj(edge_index, n)
        
        if self.center:
            x = x - x.mean(dim=0)

        # Compute F = inv(\psi) * x
        invphi_x = self._approx_f(A, x)

        # Compute orthonormal W
        _, eig_vec = torch.linalg.eigh(x.t().mm(invphi_x))
        self.weight.data = eig_vec[:, -self.c_out:]

        # Non-linear activation
        out = self.act(invphi_x.matmul(self.weight) + self.bias)
        return out

    def freeze(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def _get_sparse_adj(self, edge_index, n):
        """Get sym. normalized adj (sparse format)"""
        row, col = edge_index
        A = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        A = A.set_diag()
        D = A.sum(dim=1).to(torch.float)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        return D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(-1, 1) * A
        
    def _approx_f(self, A, x):
        """Iterative approx. of F ~ inv(I + \alpha*L) * x"""
        invphi_x = x
        for _ in range(self.niter):
            AF = A.matmul(invphi_x)
            invphi_x = self.alpha/(1+self.alpha)*AF + 1/(1+self.alpha)*x
        return invphi_x
            

        
