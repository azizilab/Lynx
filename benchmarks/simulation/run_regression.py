"""Regression baseline (ncem-style linear model): per receiver type, every gene's log-normalised expression on the counts
of each other type within the graph radius; t test per (sender, gene), BH within (receiver, sender).
    python run_regression.py <tissue dir> <out jsonl> [--radius 50]
"""
import os, sys, glob, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
from scipy import stats
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "util"))
from test_assoc import bh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissues"); ap.add_argument("out"); ap.add_argument("--radius", type=float, default=50.0)
    a = ap.parse_args()
    done = {json.loads(l)["tissue"] for l in open(a.out)} if os.path.exists(a.out) else set()
    for f in sorted(glob.glob(os.path.join(a.tissues, "*.h5ad"))):
        name = os.path.basename(f)[:-5]
        if name in done:
            continue
        t0 = time.time()
        ad = sc.read_h5ad(f)
        sc.pp.normalize_total(ad); sc.pp.log1p(ad)
        X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
        ct = ad.obs["cell_type"].astype(str).values; types = sorted(set(ct)); T = len(types); G = X.shape[1]
        lab = np.array([types.index(c) for c in ct])
        xy = np.asarray(ad.obsm["spatial"], float)
        nb = cKDTree(xy).query_ball_point(xy, r=a.radius)
        E = np.zeros((len(xy), T))
        for i, js in enumerate(nb):
            for j in js:
                if j != i:
                    E[i, lab[j]] += 1
        q = np.full((T, T, G), np.nan); beta = np.full((T, T, G), np.nan); tt = np.full((T, T, G), np.nan)
        for b in range(T):
            I = np.flatnonzero(lab == b); others = [x for x in range(T) if x != b]
            D = np.column_stack([np.ones(len(I)), E[I][:, others]]); y = X[I]
            keep = D.std(0) > 0; keep[0] = True
            Dk = D[:, keep]
            coef, _, rank, _ = np.linalg.lstsq(Dk, y, rcond=None)
            resid = y - Dk @ coef; df = max(len(I) - Dk.shape[1], 1)
            s2 = (resid ** 2).sum(0) / df
            XtX_inv = np.linalg.pinv(Dk.T @ Dk)
            se = np.sqrt(np.outer(np.diag(XtX_inv), s2))
            with np.errstate(invalid="ignore", divide="ignore"):
                tv = coef / se
            p = 2 * stats.t.sf(np.abs(tv), df)
            col = 1
            for k_, aa in enumerate(others):
                if not keep[k_ + 1]:
                    continue
                pa = p[col]; ok = np.isfinite(pa); qa = np.ones(G)
                if ok.any():
                    qa[ok] = bh(pa[ok])
                q[b, aa] = qa; beta[b, aa] = coef[col]; tt[b, aa] = tv[col]; col += 1
        rec = dict(tissue=name, types=types, genes=list(ad.var_names), radius=a.radius,
                   q=np.round(q, 6).tolist(), beta=np.round(beta, 6).tolist(), t=np.round(tt, 4).tolist())
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
