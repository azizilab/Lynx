"""SVCA (Arnol, Schapiro, Bodenmiller, Saez-Rodriguez & Stegle 2019, Cell Reports 29:202; github.com/damienArnol/svca)
as a baseline on the exported simulated tissues (export_tissues.py), re-implemented (the published code needs a bundled
2019 limix) and resolved by cell type, which the receiver C x C / C x C x G read-out needs:

    python run_SVCA.py <tissue dir> <out jsonl> [--worker K --n-workers N] [--boot 40] [--min-det 0.05] [--pcs 20]
"""
import os, sys, glob, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata, chi2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "util"))
from test_assoc import bh  # noqa: E402

L_GRID = [10.0, 20.0, 40.0, 80.0, 150.0]
torch.set_default_dtype(torch.float64)


def preprocess(X):
    """Variance stabilisation log(x + 1/(2 phi)) per gene, the log total counts regressed out."""
    mu, var = X.mean(0), X.var(0)
    phi = np.clip((var - mu) / np.maximum(mu, 1e-8) ** 2, 1e-3, 1e3)
    Y = np.log(X + 1.0 / (2.0 * phi[None]))
    lt = np.log(np.maximum(X.sum(1), 1.0))
    D = np.column_stack([np.ones(len(lt)), lt])
    return Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]


def qnorm(y):
    return norm.ppf((rankdata(y) - 0.5) / len(y))


class LRGP:
    """y ~ N(0, s_e E + s_n I + sum_k s_k W_k W_k^T) with E = Q diag(lam) Q^T, all in E's eigenbasis (yt = Q^T y,
    Wt_k = Q^T W_k); E and every W_k W_k^T Gower-normalised, so the variance fractions are the s's over their sum."""

    def __init__(self, yt, lam, Wt):
        self.y = torch.as_tensor(yt); self.lam = torch.as_tensor(lam); self.W = [torch.as_tensor(w) for w in Wt]
        self.n = len(yt)

    def nll(self, th, active):
        d = torch.exp(th[0]) * self.lam + torch.exp(th[1])
        Dinv = 1.0 / d
        quad = (self.y * self.y * Dinv).sum()
        logdet = torch.log(d).sum()
        if active:
            Ws = torch.cat([torch.exp(0.5 * th[2 + k]) * self.W[k] for k in active], 1)
            M = torch.eye(Ws.shape[1]) + Ws.T @ (Ws * Dinv[:, None])
            L = torch.linalg.cholesky(M)
            v = Ws.T @ (self.y * Dinv)
            u = torch.cholesky_solve(v[:, None], L)[:, 0]
            quad = quad - v @ u
            logdet = logdet + 2 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * quad + 0.5 * logdet + 0.5 * self.n * np.log(2 * np.pi)

    def fit(self, active, th0=None):
        m = len(self.W)
        th = np.full(2 + m, -12.0)
        if th0 is not None:
            th[:] = th0
        else:
            init = np.log(1.0 / (len(active) + 2)); th[0] = th[1] = init; th[[2 + k for k in active]] = init
        free = [0, 1] + [2 + k for k in active]
        idx = torch.tensor(free); base = torch.tensor(th)

        def f(v):
            tv = torch.tensor(v, requires_grad=True)
            val = self.nll(base.index_put((idx,), tv), active)
            val.backward()
            return float(val), tv.grad.numpy().copy()
        r = minimize(f, th[free], jac=True, method="L-BFGS-B", bounds=[(-12.0, 3.0)] * len(free), options=dict(maxiter=300))
        th[free] = r.x
        return th, -float(r.fun)


def fractions(th, active):
    s = {k: np.exp(th[2 + k]) for k in active}
    tot = np.exp(th[0]) + np.exp(th[1]) + sum(s.values())
    return {k: v / tot for k, v in s.items()}, np.exp(th[0]) / tot, np.exp(th[1]) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissues"); ap.add_argument("out")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--boot", type=int, default=40); ap.add_argument("--min-det", type=float, default=0.05)
    ap.add_argument("--pcs", type=int, default=20); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-genes", type=int, default=0, help="smoke tests: genes per receiver (0 = all)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = a.out.replace(".jsonl", f"_{a.worker}.jsonl") if a.n_workers > 1 else a.out
    done = {json.loads(l)["tissue"] for l in open(out)} if os.path.exists(out) else set()
    files = sorted(glob.glob(os.path.join(a.tissues, "*.h5ad")))
    rng = np.random.default_rng(11)
    for kf, f in enumerate(files):
        if kf % a.n_workers != a.worker:
            continue
        name = os.path.basename(f)[:-5]
        if name in done:
            continue
        t0 = time.time()
        ad = sc.read_h5ad(f)
        Xc = np.asarray(ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X, float)
        Y = preprocess(Xc)
        xy = np.asarray(ad.obsm["spatial"], float)
        types = sorted(ad.obs["cell_type"].astype(str).unique())
        ct = np.array([types.index(c) for c in ad.obs["cell_type"].astype(str)])
        T, G = len(types), Xc.shape[1]
        Yc = Y - Y.mean(0)
        V = np.linalg.svd(Yc, full_matrices=False)[2][:a.pcs].T                  # gene loadings (G, r)
        frac = np.full((T, T, G), np.nan); P = np.full((T, T, G), np.nan); Q = np.full((T, T, G), np.nan)
        other = {k: np.full((T, G), np.nan) for k in ("intrinsic", "environment", "noise")}
        meta = {}
        for b in range(T):
            rb = np.where(ct == b)[0]; nb = len(rb)
            senders = [s for s in range(T) if s != b]
            det = (Xc[rb] > 0).mean(0)
            testable = np.where(det >= a.min_det)[0]
            if a.max_genes:
                testable = testable[:a.max_genes]
            Xb = Y[rb] - Y[rb].mean(0)
            Ub = Xb @ V
            Db = cdist(xy[rb], xy[rb])
            Xs = {s: Y[ct == s] - Y[ct == s].mean(0) for s in senders}
            Dbs = {s: cdist(xy[rb], xy[ct == s]) for s in senders}
            prep = {}

            def setup(l):
                if l in prep:
                    return prep[l]
                E = np.exp(-Db ** 2 / (2 * l * l))
                E /= (np.trace(E) - E.sum() / nb) / (nb - 1)
                lam, Qe = np.linalg.eigh(E)
                lam = np.clip(lam, 0.0, None)
                q1 = Qe.T @ np.ones(nb)
                QtU, QtX = Qe.T @ Ub, Qe.T @ Xb
                QtH, QtZX = {}, {}
                for s in senders:
                    Z = np.exp(-Dbs[s] ** 2 / (2 * l * l))
                    ZX = Z @ Xs[s]
                    QtZX[s] = Qe.T @ ZX
                    QtH[s] = QtZX[s] @ V
                prep[l] = (Qe, lam, q1, QtU, QtX, QtH, QtZX)
                return prep[l]

            def blocks(g, l):
                Qe, lam, q1, QtU, QtX, QtH, QtZX = setup(l)
                Ws = [QtU - np.outer(QtX[:, g], V[g])]                              # intrinsic, focal gene removed
                Ws += [QtH[s] - np.outer(QtZX[s][:, g], V[g]) for s in senders]      # interaction with each sender type
                out_ = []
                for W in Ws:
                    c = ((W ** 2).sum() - ((q1 @ W) ** 2).sum() / nb) / (nb - 1)
                    out_.append(W / np.sqrt(max(c, 1e-12)))
                return Qe, lam, out_

            def model(g, l, y=None):
                Qe, lam, Ws = blocks(g, l)
                y = qnorm(Xb[:, g]) if y is None else y
                return LRGP(Qe.T @ y, lam, Ws), Qe, lam, Ws

            var_order = testable[np.argsort(-Xb[:, testable].var(0))][:20]
            best_l, best_ll = L_GRID[0], -np.inf
            for l in L_GRID:
                tot = 0.0
                for g in var_order:
                    gp = model(g, l)[0]
                    tot += gp.fit(list(range(len(gp.W))))[1]
                if tot > best_ll:
                    best_l, best_ll = l, tot
            l = best_l
            llr = np.full((len(senders), G), np.nan)
            fits = {}
            for g in testable:
                gp = model(g, l)[0]
                act = list(range(len(gp.W)))
                th, ll_full = gp.fit(act)
                fr, fe, fn = fractions(th, act)
                other["intrinsic"][b, g], other["environment"][b, g], other["noise"][b, g] = fr[0], fe, fn
                for k_s, s in enumerate(senders):
                    frac[s, b, g] = fr[1 + k_s]
                    if fr[1 + k_s] < 1e-4:
                        llr[k_s, g] = 0.0; continue
                    red = [k for k in act if k != 1 + k_s]
                    th0 = th.copy(); th0[2 + 1 + k_s] = -12.0
                    llr[k_s, g] = max(2.0 * (ll_full - gp.fit(red, th0=th0)[1]), 0.0)
                fits[g] = th
            null = []
            for _ in range(a.boot):
                g = int(rng.choice(testable)); k_s = int(rng.integers(len(senders)))
                gp, Qe, lam, Ws = model(g, l)
                th = fits[g].copy(); th[2 + 1 + k_s] = -12.0
                yt = np.sqrt(np.exp(th[0]) * lam + np.exp(th[1])) * rng.normal(size=nb)
                for k in range(len(Ws)):
                    if k != 1 + k_s:
                        yt = yt + np.exp(0.5 * th[2 + k]) * (Ws[k] @ rng.normal(size=Ws[k].shape[1]))
                ysim = qnorm(Qe @ yt)
                gps = LRGP(Qe.T @ ysim, lam, Ws)
                act = list(range(len(Ws)))
                thf, llf = gps.fit(act)
                fr, _, _ = fractions(thf, act)
                if fr[1 + k_s] < 1e-4:
                    null.append(0.0); continue
                red = [k for k in act if k != 1 + k_s]
                th0 = thf.copy(); th0[2 + 1 + k_s] = -12.0
                null.append(max(2.0 * (llf - gps.fit(red, th0=th0)[1]), 0.0))
            null = np.array(null)
            pos = null[null > 1e-8]
            p0 = 1.0 - (len(pos) + 1.0) / (len(null) + 2.0)      # the null's point mass at 0, smoothed (never exactly 1)
            if len(pos) >= 5 and pos.var() > 0:
                scale, dfree = pos.var() / (2 * pos.mean()), 2 * pos.mean() ** 2 / pos.var()
            else:
                scale, dfree = 1.0, 1.0
            meta[types[b]] = dict(length_scale=l, null_n=int(len(null)), null_zero_mass=float(p0), null_scale=float(scale),
                                  null_df=float(dfree), n_cells=int(nb), n_testable=int(len(testable)))
            for k_s, s in enumerate(senders):
                v = llr[k_s]
                p = np.where(np.isfinite(v), np.where(v > 1e-8, (1 - p0) * chi2.sf(v / scale, dfree), 1.0), np.nan)
                P[s, b] = p
                ok = np.isfinite(p)
                if ok.any():
                    Q[s, b, ok] = bh(p[ok])
            print(name, "receiver", types[b], f"n {nb} genes {len(testable)} l {l} null zero-mass {p0:.2f} chi2 scale {scale:.2f} df {dfree:.2f}",
                  f"{time.time() - t0:.0f}s", flush=True)
        r6 = lambda A: np.where(np.isfinite(A), np.round(A, 6), np.nan).tolist()
        rec = dict(tissue=name, types=types, genes=list(ad.var_names), frac=r6(frac), p=r6(P), q=r6(Q),
                   other={k: r6(v) for k, v in other.items()}, meta=meta, pcs=a.pcs, sec=round(time.time() - t0, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
