import numpy as np
import networkx as nx

from scipy.sparse import linalg
from skimage.segmentation import find_boundaries
from __init__ import LOGGER


class HeatDiffusion:
    r"""Generate noisy estimate of zonation trajectory 
    from Graph-based heat diffusion w/ dirichlet constraints
    on U(cv):=1 & U(pv):=-1
    """
    def __init__(
        self,
        vein_prior: np.ndarray,
        roi: np.ndarray = None,
        cv_val: int = 1,
        pv_val: int = -1,
        ndim: int = 2,
        anis: float = 10.0,
    ):
        self.shape = vein_prior.shape
        self.ndim = ndim
        self.weight_xy = 1.0
        self.weight_z = anis * self.weight_xy

        self.cv_coords = np.where(vein_prior == cv_val)
        self.pv_coords = np.where(vein_prior == pv_val)
        self.cv_nodes = {tuple(idx)
                         for idx in np.array(self.cv_coords).T}
        self.pv_nodes = {tuple(idx)
                         for idx in np.array(self.pv_coords).T}
        self.roi = np.ones_like(vein_prior) if roi is None else roi
        
        LOGGER.info("Creating {0}D graph w/ dimension {1}...".format(ndim, self.roi.shape))
        self.G = self._create_graph()

        LOGGER.info("Initializing boundary temperature `U_b`...")
        self._set_constraints()

        self.U_i = None  # Interior node temperature
        self.U = None   # whole-slide temperature in the image space
        self.interior_nodes = None

    def _create_graph(self):
        r"""Convert combinatory graph w/ 4-connected components (2D)
        or 6-connected components (3D) within ROI pixels
        """        
        if self.ndim == 2:
            # Edge along X & Y axes
            hx, hy =  np.nonzero(np.logical_and(self.roi[1:] == 1, self.roi[:-1] == 1)) 
            hpos = [hx, hy]
            hshift = (1, 0)

            vx, vy = np.nonzero(np.logical_and(self.roi[:,1:] == 1, self.roi[:,:-1] == 1))  
            vpos = [vx, vy]
            vshift = (0, 1)
        elif self.ndim == 3:
            # Edge along X, Y & Z axes
            hx, hy, hz = np.nonzero(np.logical_and(self.roi[1:] == 1, self.roi[:-1] == 1))  
            hpos = [hx, hy, hz]
            hshift = (1, 0, 0)
            
            vx, vy, vz = np.nonzero(np.logical_and(self.roi[:,1:,:] == 1, self.roi[:,:-1,:] == 1))  
            vpos = [vx, vy, vz]
            vshift = (0, 1, 0)

            ux, uy, uz = np.nonzero(np.logical_and(self.roi[:,:,1:] == 1, self.roi[:,:,:-1] == 1)) 
            upos = [ux, uy, uz]
            ushift = (0, 0, 1)
        else:
            raise ValueError("Only support 2D / 3D heat diffusion")

        h_units = np.array(hpos).T
        h_starts = [tuple(n) for n in h_units]
        h_ends = [tuple(n) for n in h_units + hshift] 
        horizontal_edges = zip(h_starts, h_ends)
        
        v_units = np.array(vpos).T
        v_starts = [tuple(n) for n in v_units]
        v_ends = [tuple(n) for n in v_units + vshift] 
        vertical_edges = zip(v_starts, v_ends)

        G = nx.Graph()
        G.add_edges_from(horizontal_edges, weight=self.weight_xy)
        G.add_edges_from(vertical_edges, weight=self.weight_xy)

        if self.ndim == 3:
            u_units = np.array(upos).T
            u_starts = [tuple(n) for n in u_units]
            u_ends = [tuple(n) for n in u_units + ushift]
            inplane_edges = zip(u_starts, u_ends)

            G.add_edges_from(inplane_edges, weight=self.weight_z)

        return G
    
    def _set_constraints(self):
        r"""Initialize temp. & ROI boundary as graph properties"""
        nadj = 4 if self.ndim == 2 else 6
        for n in self.G:
            if n in self.cv_nodes:
                self.G.nodes[n]['t'] = 1
                self.G.nodes[n]['bound'] = True
            elif n in self.pv_nodes:
                self.G.nodes[n]['t'] = -1
                self.G.nodes[n]['bound'] = True
            else:
                self.G.nodes[n]['t'] = 0
                criteria = self.G.degree[n] < nadj if self.ndim == 2 else \
                           self.G.degree[n] < nadj and \
                           (n[1] == 0 or n[1] == self.shape[1]-1 or n[2] == 0 or n[2] == self.shape[2]-1)
                if criteria:
                    self.G.nodes[n]['bound'] = True
        return None
    
    def get_interior_U(self, debug=False):
        r"""Compute temperature of "interior" nodes based on 
        Harmonic interpolation solution (Grady & Schwartz, 2003)
        """
        LOGGER.info("Inferring `interior node` temperature `U_i`...")

        # Construct permuted Laplacian Matrix L => {L_b, L_i, R, R^T}
        bound_nodes = [n for n, v in self.G.nodes.items()
                       if 'bound' in v]
        interior_nodes = [n for n, v in self.G.nodes.items()
                          if 'bound' not in v]

        n_bound = len(bound_nodes)
        perm_node_orders = bound_nodes + interior_nodes
        if debug:
            assert len(self.G) == len(perm_node_orders)

        L = nx.laplacian_matrix(self.G, nodelist=perm_node_orders)
        L_i = L[n_bound:, n_bound:]
        R = L[:n_bound, n_bound:]

        # Validate permuted nodes' in-degree have the correct order [d(bound), d(interior)]
        if debug:
            diag = np.diag(L)
            for i, n in enumerate(perm_node_orders):
                assert self.G.degree[n] == diag[i]

        # Compute interior temperature u(i) from L & u(b): 
        # Sol:  L_i @ u(i) = -R^T @ u(b)
        U_b = np.asarray([self.G.nodes[n]['t'] for n in bound_nodes])
        U_i = linalg.cg(A=L_i, b=-R.T@U_b) 

        if isinstance(U_i, tuple):
            U_i = U_i[0]

        self.U_i = U_i
        self.interior_nodes = tuple(np.array(interior_nodes).T) # 2 x N tuple  
        return self.U_i, self.interior_nodes  
    
    def infer_zone_dynamics(self):
        r"""Assign combinatorial steady-state solutions of the 
        diffused tempeture (U) back to the original image space
        """
        LOGGER.info("Projecting temperature {U_b, U_i} back to image space...")
        assert self.U_i is not None, "Please infer interior node tempeture first"
        assert len(self.interior_nodes[0]) == len(self.U_i), 'Different coords & temperature lengths'
        
        self.U = np.zeros(self.shape, dtype=np.float64)
        self.U[self.interior_nodes] = self.U_i
        self.U[self.cv_coords] = 1
        self.U[self.pv_coords] = -1
        return self.convert_gradients(self.U)
    
    def infer_zones(
        self,
        n_layers=10,
        return_border=False,
        verbose=False
    ):
        r"""Create discretized 1-indexed bins (1,2,...,n) as the zonation estimates
        from diffused gradient temperature `u`, keep CV & PV regions off from `roi` 
        as the min (PV) / max (CV) zones
        """
        LOGGER.info("Predicting discretized lobule layers (zonations)...")
        assert self.U is not None, "Please compute temperature (U) in the image space first"
        assert n_layers > 3, "Invalid `n_layers`, please assign # lobule layers > 3"
        
        coords = np.nonzero(self.roi)
        coords_to_rm = np.nonzero(1-self.roi)
        qs = np.quantile(self.U[coords], np.linspace(0, 1, n_layers-1))

        if verbose:
            print('Quantile:', qs)
            
        zone = np.zeros_like(self.U, dtype=np.int32)
        for i, q in enumerate(qs[:-1]):
            zone[self.U >= q] = i+1
        zone[coords_to_rm] = 0

        zone[self.cv_coords] = zone.max() + 1
        zone += 1

        # Assign 1-pixel width border btw zones
        if return_border:
            border = find_boundaries(zone)
            zone[border] = 0

        return zone
    
    @staticmethod
    def convert_gradients(gradients):
        r"""convert gradients to [0-1]"""
        v = gradients + gradients.min()
        return (v-v.min()) / (v.max()-v.min())