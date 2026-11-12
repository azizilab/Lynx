"""Accuracy of the kappa posterior solve on a fitted run: the tissue bound (the steps that guarantee `tol` for any weights the
model can learn on the graph: cond(H) <= 1 + (2 + sqrt(D))^2 / lambda, D the most receivers of one sender), the steps the
certified solve used, and its true error against a 3000-step reference, relative to |kappa* - mu|.

    python kappa_solve.py breast|liver <state> [--tol 0.01] [--data DIR]    (state: the best_model.pth of run_Lynx.py's fit)
"""
import os, sys, argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch_scatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, f"{ROOT}/models", f"{ROOT}/util"]
import configs, dataset, vgae, IO  # noqa: E402


@torch.no_grad()
def check(m, data, tol=0.01, ref_steps=3000):
    x = m.lognorm(data[m.ref].x)
    eid, n = data.edge_index_dict, x.size(0)
    u = data[m.query].x if m.patch_size < 0 else m._reshape_patches(data[m.query].x)
    s_q, d_q = eid[m.q2r]
    z = torch_scatter.scatter_mean(m.encode_z(x, u, eid)[0][s_q], d_q, dim=0, dim_size=n)
    ei = eid[m.r2r]
    omega = m._guide_omega(x, z, eid, data.edge_attr_dict[m.r2r])
    s_hat, cl = m.encode_kappa(x)[0], data[m.ref].cluster
    m.configs.kappa_solve_tol, m.configs.kappa_solve_iters = tol, 0
    m.kappa_solve_stats = dict(solves=0, steps=0, max_steps=0, uncertified=0)
    k_cert = m._kappa_solve(ei, omega, z, s_hat, cl)
    stats, cap = dict(m.kappa_solve_stats), m._kappa_solve_cap(ei, n, tol)
    m.configs.kappa_solve_tol, m.configs.kappa_solve_iters = 0.0, ref_steps
    k_ref = m._kappa_solve(ei, omega, z, s_hat, cl)
    mu = m.kappa_mu(cl)
    D = int(torch.bincount(ei[0], minlength=n).max())
    return dict(D=D, bound_steps=cap, steps=stats["steps"], uncertified=stats["uncertified"],
                error=float((k_cert - k_ref).norm() / (k_ref - mu).norm()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tissue"); ap.add_argument("state"); ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.tissue == "breast":  # the inputs of breast/run_Lynx.py
        path = a.data or os.path.join(ROOT, "data/breast/dcis_fov")
        xen, he = sc.read_h5ad(os.path.join(path, "cell_feature_matrix.h5")), sc.read_h5ad(os.path.join(path, "he_patches_norm.h5ad"))
        ct = xen.obs["cell_type"].astype(str)
        keep = (ct != "Unlabeled") & ~ct.isin(ct.value_counts()[ct.value_counts() < 10].index) & ~ct.str.contains("Hybrid", case=False)
        xen.obs["cell_type"] = ct.replace({"DCIS_1": "DCIS", "Prolif_Invasive_Tumor": "Invasive_Tumor"}).astype("category")
        xen, he = xen[keep.values].copy(), he[keep.values].copy()
        gd = dataset.HeteroDataset(adatas_ref=xen, adatas_query=he, n_subgraphs=1, k=8, r=50, is_weighted=True, cluster_key="cell_type",
                                   alpha=0.5, query="HE", query_proj_key="spatial", ref="Xenium", ref_proj_key="spatial")
        mc = configs.set_model_configs(graph_data=gd, c_hidden=32, c_latent=6, patch_size=int(np.sqrt(he.n_vars // 3)), act=nn.SiLU())
    else:  # the inputs of liver/run_Lynx.py
        path = a.data or os.path.join(ROOT, "data")
        xen = IO.load_xenium(os.path.join(path, "xenium", "NIH_F5_proseg"), load_img=False)
        xen, desi = IO.filter_cells(xen, sc.read_h5ad(os.path.join(path, "desi", "NIH_F5_proseg.h5")), by="map")
        gd = dataset.HeteroDataset(adatas_ref=xen, adatas_query=desi, n_subgraphs=1, r=50, is_weighted=True, alpha=0.5, cluster_key="subtype")
        mc = configs.set_model_configs(graph_data=gd, c_hidden=32, c_latent=6, act=nn.SiLU())
    m = vgae.HeteroAttnVGAE(mc, device=dev).to(dev)
    m.load_state_dict(torch.load(a.state, map_location=dev))
    print(a.state, check(m.eval(), gd[0].to(dev), a.tol))


if __name__ == "__main__":
    main()
