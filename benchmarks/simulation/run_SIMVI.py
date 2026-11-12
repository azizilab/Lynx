"""SIMVI (Dong et al., Nat Commun 2025; package simvi 0.1.2) on the simulation as the paper prescribes (k-NN graph k = 10,
SimVIModel defaults, the MERFISH benchmark's training: 100 epochs, batch 500, 25 permutation-pretraining epochs, lr 1e-3);
--config default: the package's training defaults. The configuration with the lower median validation loss is used.

    ~/future/envs/simvi/bin/python run_SIMVI.py "<h5ad glob>" <outdir> [--config paper|default]
"""
import os, glob, time, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from scipy.stats import f as fdist
from sklearn.linear_model import LinearRegression


def se_individual(Xlog, Zc, s, num_arch, seed, Kfold=5, delta=0.1, maxiter=200, eps=0.0):
    """get_se(mode='individual') with covariates Zc (z + one-hot type): per-archetype SE, the summed SE and the residuals."""
    from py_pcha import PCHA
    np.random.seed(seed)
    XC, S, C, SSE, varexpl = PCHA(s.T, noc=num_arch, delta=delta, conv_crit=1e-5, maxiter=maxiter, verbose=False)
    S = np.asarray(S.T)
    np.random.seed(42); idx = np.random.permutation(Zc.shape[0]); split = np.array_split(idx, Kfold)
    N, G = Xlog.shape; se_tot = np.zeros((N, G)); Xres = np.zeros((N, G)); se_arch = []
    for i in range(S.shape[1]):
        se = np.zeros((N, G))
        for ind in split:
            Si = S[ind, i]
            lr = LinearRegression().fit(Zc[ind], Si); lr2 = LinearRegression().fit(Zc[ind], Xlog[ind])
            Si_ = Si - lr.predict(Zc[ind]); X_ = Xlog[ind] - lr2.predict(Zc[ind])
            lr3 = LinearRegression().fit(Zc[ind], X_ / (Si_[:, None] + eps), sample_weight=Si_ ** 2)
            se[ind] = lr3.predict(Zc[ind]) * Si_[:, None]; Xres[ind] = X_
        se_arch.append(se); se_tot += se
    return se_tot, Xres, se_arch, S, float(varexpl)


def per_type(se_tot, Xres, ct, T, k_one, n_arch):
    """Within-type R2 of the summed SE on the residualised expression, and its F-test p (regressors k_one x n_arch)."""
    G = se_tot.shape[1]; R2 = np.full((T, G), np.nan); P = np.full((T, G), np.nan); n = np.zeros(T, int)
    for b in range(T):
        sel = ct == b; n[b] = sel.sum()
        if n[b] < 30:
            continue
        y, e = Xres[sel], se_tot[sel]; sst = ((y - y.mean(0)) ** 2).sum(0); ssr = ((y - e) ** 2).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(sst > 0, 1 - ssr / sst, np.nan)
            dfm = k_one * n_arch; dfr = n[b] - dfm - 1
            Fv = (np.clip(r2, 0, None) / dfm) / (np.clip(1 - r2, 1e-12, None) / dfr)
        R2[b], P[b] = r2, np.where(np.isfinite(r2), fdist.sf(Fv, dfm, dfr), np.nan)
    return R2, P, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files"); ap.add_argument("outdir"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--config", default="paper", choices=["paper", "default"])
    a_ = ap.parse_args(); os.makedirs(a_.outdir, exist_ok=True)
    import scvi
    from simvi.model.simvi import SimVIModel
    for k, f in enumerate(sorted(glob.glob(a_.files))):
        name = os.path.basename(f).replace(".h5ad", ""); out = os.path.join(a_.outdir, f"simvi__{name}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time(); seed = int(name.split("_")[1])
        A = ad.read_h5ad(f); A.X = np.asarray(A.X.toarray() if hasattr(A.X, "toarray") else A.X, np.float32)
        A.obs["cell_type"] = A.obs["cell_type"].astype(str).astype("category")
        ct = np.array([int(c.replace("type", "")) for c in A.obs["cell_type"]]); T = int(ct.max()) + 1
        scvi.settings.seed = seed; torch.manual_seed(seed); np.random.seed(seed)
        SimVIModel.setup_anndata(A)
        edge_index = SimVIModel.extract_edge_index(A, spatial_key="spatial", n_neighbors=10)
        model = SimVIModel(A)
        if a_.config == "paper":
            tl, vl = model.train(edge_index, max_epochs=100, batch_size=500, mae_epochs=25, use_gpu=True)
        else:
            # full batch as one mini-batch of every cell (identical computation: the package's mini-batch path masks the same
            # full-graph inference to the same training cells); its batch_size=None path calls numpy on a GPU loss tensor
            tl, vl = model.train(edge_index, use_gpu=True, batch_size=A.n_obs)
        ei = edge_index.to(next(model.module.parameters()).device)          # train() moves only its own copy of the graph
        z = model.get_latent_representation(ei, representation_kind="intrinsic")
        s = model.get_latent_representation(ei, representation_kind="interaction")
        B = A.copy(); sc.pp.normalize_total(B); sc.pp.log1p(B); Xlog = np.asarray(B.X, float)
        onehot = pd.get_dummies(A.obs["cell_type"]).values.astype(float); Zc = np.hstack([z, onehot])
        arrs = dict(z=z.astype(np.float32), s=s.astype(np.float32), train_loss=np.asarray(tl, float), val_loss=np.asarray(vl, float), ct=ct.astype(np.int16))
        for na in (5, 7):
            se_tot, Xres, se_arch, S, ve = se_individual(Xlog, Zc, s, na, seed)
            R2, P, n = per_type(se_tot, Xres, ct, T, z.shape[1] + 1, na)
            arrs.update({f"se_a{na}": se_tot.astype(np.float32), f"r2type_a{na}": R2.astype(np.float32), f"ptype_a{na}": P, f"arch_varexpl_a{na}": np.array(ve)})
            if a_.check and k == 0 and na == 5:                      # the package's own get_se, same archetype initialisation
                np.random.seed(seed)
                se_list, r2z, r2s, pz, ps, S2 = model.get_se(ei, cell_type_label="cell_type", num_arch=5)
                diff = float(np.max(np.abs(np.sum(se_list, 0) - se_tot))); print("check vs get_se: max |SE diff|", diff, "| scale", float(np.abs(se_tot).max()), flush=True)
                arrs["check_maxdiff"] = np.array(diff)
        np.savez_compressed(out, **arrs)
        print(name, f"{time.time() - t0:.0f}s", "| final train / val loss", round(float(tl[-1]), 2), round(float(vl[-1]), 2), flush=True)


if __name__ == "__main__":
    main()
