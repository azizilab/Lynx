"""Niche-DE (Mason et al. 2024, Genome Biology; pure-Python port `pynichede` 0.1.0 of the R package) as a baseline on
the exported simulated tissues (export_tissues.py). Runs in the amici environment:

    python -I run_NicheDE.py <tissue dir> <out jsonl> [--sigma 1,100,250] [--cores 4] [--C 150 --M 10 --gamma 0.8]
"""
import os, glob, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm, cauchy


def interaction_level(obj, n_ct, ngene):
    from pynichede.rstats import p_adjust
    K = len(obj.sigma)
    ll = np.array([[r["log_likelihood"] for r in obj.niche_DE[k]] for k in range(K)], float).T    # gene x kernel
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(np.where(np.isnan(ll), -np.inf, ll), axis=1)
        W = np.exp(ll - mx[:, None])
    W[~np.isfinite(W)] = 0.0
    W[W < 0.1] = 0.0
    out = {}
    beta = np.zeros((n_ct, n_ct, ngene)); bden = np.zeros((n_ct, n_ct, ngene))
    for pos, lab in ((True, "pos"), (False, "neg")):
        num = np.zeros((n_ct, n_ct, ngene)); den = np.zeros((n_ct, n_ct, ngene))
        for k in range(K):
            for g, r in enumerate(obj.niche_DE[k]):
                if r["T_stat"] is None or W[g, k] == 0:
                    continue
                T = np.asarray(r["T_stat"], float)
                pk = 1.0 - norm.cdf(T if pos else -T)                       # its T_to_p 'positive' / 'negative'
                t = np.tan((0.5 - pk) * np.pi); ok = np.isfinite(t)
                num[:, :, g][ok] += W[g, k] * t[ok]; den[:, :, g][ok] += W[g, k]
                if pos:
                    B = np.asarray(r["betas"], float); okb = np.isfinite(B)
                    beta[:, :, g][okb] += W[g, k] * B[okb]; bden[:, :, g][okb] += W[g, k]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 1.0 - cauchy.cdf(num / den)
        p[den == 0] = np.nan
        padj = p.copy()
        for g in range(ngene):
            for i in range(n_ct):
                fin = np.isfinite(p[i, :, g])
                if fin.any():
                    padj[i, fin, g] = p_adjust(p[i, fin, g], "BH")
        out[lab] = dict(raw=p, adj=padj)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = beta / bden
    beta[bden == 0] = np.nan
    return out, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissues"); ap.add_argument("out")
    ap.add_argument("--sigma", default="1,100,250")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--C", type=float, default=150); ap.add_argument("--M", type=int, default=10); ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    import pynichede as nde
    sigma = [float(x) for x in a.sigma.split(",")]
    out = a.out.replace(".jsonl", f"_{a.worker}.jsonl") if a.n_workers > 1 else a.out
    done = {json.loads(l)["tissue"] for l in open(out)} if os.path.exists(out) else set()
    files = sorted(glob.glob(os.path.join(a.tissues, "*.h5ad")))
    for k, f in enumerate(files):
        if k % a.n_workers != a.worker:
            continue
        name = os.path.basename(f)[:-5]
        if name in done:
            continue
        t0 = time.time()
        adata = sc.read_h5ad(f)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        counts = pd.DataFrame(np.rint(X).astype(int), index=adata.obs_names, columns=adata.var_names)
        ct = adata.obs["cell_type"].astype(str).values
        types = sorted(set(ct))
        coord = pd.DataFrame(np.asarray(adata.obsm["spatial"], float), index=adata.obs_names, columns=["x", "y"])
        lib = nde.create_library_matrix(counts, pd.DataFrame({"cell": adata.obs_names, "cell_type": ct}), random_state=0)
        deconv = pd.DataFrame(np.array([[1.0 if c == t_ else 0.0 for t_ in types] for c in ct]), index=adata.obs_names, columns=types)
        lib = lib.loc[types]
        obj = nde.create_nichede_object(counts, coord, lib, deconv, sigma=sigma)
        obj = nde.calculate_effective_niche(obj, cutoff=0.05)
        obj = nde.niche_DE(obj, num_cores=a.cores, C=a.C, M=a.M, gamma=a.gamma, Int=True, batch=True, self_EN=False, verbose=False)
        gn, cts = [str(g) for g in obj.gene_names], [str(c) for c in obj.cell_types]
        inter, beta = interaction_level(obj, len(cts), len(gn))
        valid = [int(sum(r["valid"] for r in obj.niche_DE[k])) for k in range(len(sigma))]
        rec = dict(tissue=name, types=cts, genes=gn, sigma=sigma, C=a.C, M=a.M, gamma=a.gamma, valid_genes_per_kernel=valid,
                   beta=np.round(beta, 5).tolist())
        for pos, lab in ((True, "pos"), (False, "neg")):
            pv = obj.niche_DE_pval_pos if pos else obj.niche_DE_pval_neg
            rec[lab] = dict(gene=np.round(np.asarray(pv["gene_level"], float), 6).tolist(),
                            cell_type=np.round(np.asarray(pv["cell_type_level"], float), 6).tolist(),
                            interaction=np.round(inter[lab]["adj"], 6).tolist(),
                            interaction_raw=np.round(inter[lab]["raw"], 6).tolist())
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, f"{time.time() - t0:.0f}s", "valid/kernel", valid, flush=True)


if __name__ == "__main__":
    main()
