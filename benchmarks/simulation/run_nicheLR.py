"""niche-LR (Mason et al. 2024, Genome Biology; the Niche-DE package's ligand-receptor step, pynichede 0.1.0 port of
nicheDE::niche_LR_cell) on the exported tissues. Runs in the amici environment:

    python -I run_nicheLR.py <tissue dir> <out jsonl> [--prior noisy,oracle] [--cores 2] [--worker K --n-workers N]
"""
import os, re, glob, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc

N_TARGETS, N_REAL_NOISY = 6, 2


def build_db(ch):
    rows = [(c["lig"], c["rec"], c["sender"], c["receiver"], "true" if c["true"] else c.get("lig_kind", "decoy"))
            for c in ch["channels"]]
    mk_l = {ch["lig_type"][str(g)]: g for g in ch["lig_db"] if not any(g == c["lig"] for c in ch["channels"])}
    mk_r = {ch["rec_type"][str(g)]: g for g in ch["rec_db"] if not any(g == c["rec"] for c in ch["channels"])}
    for a, gl in mk_l.items():
        for b, gr in mk_r.items():
            if a != b:
                rows.append((gl, gr, a, b, "marker"))
    return pd.DataFrame(rows, columns=["lig", "rec", "sender", "receiver", "kind"])


def build_prior(db, truth, G, seed, kind):
    rng = np.random.default_rng(10_000 + seed)
    lr = set(db["lig"]) | set(db["rec"])
    resp = {g for p in truth["pairs"] for g in p["genes"]}
    non = np.array([g for g in range(G) if g not in lr and g not in resp])
    up = {(p["sender"], p["receiver"]): [g for g, v in zip(p["genes"], p["v"]) if v > 0] for p in truth["pairs"]}
    ligs = sorted(set(db["lig"]))
    LT = rng.uniform(0, 0.05, size=(G, len(ligs)))
    for j, l in enumerate(ligs):
        row = db[(db["lig"] == l)].iloc[0]
        rand = list(rng.choice(non, size=N_TARGETS, replace=False))
        if row["kind"] == "true":
            real = up[(int(row["sender"]), int(row["receiver"]))]
            tg = real if kind == "oracle" else real[:N_REAL_NOISY] + rand[:N_TARGETS - N_REAL_NOISY]
        else:
            tg = rand
        LT[tg, j] = 1.0
    return pd.DataFrame(LT, index=[f"g{g}" for g in range(G)], columns=[f"g{l}" for l in ligs])


def ligand_activity(obj, ltm, index, niche, K=25, truncation_value=3):
    """pynichede.downstream._ligand_scores' activity for every candidate ligand (the package returns only those over
    its cut); same steps: niche-DE T of the index type vs the niche type at each ligand's best kernel, truncated, the
    ligand's top-K targets by scaled potential, weights = scaled potential / mean, sum(T x w) / sqrt(sum w^2)."""
    from pynichede.downstream import _top_kernel, _order_decreasing, _dense
    from pynichede.rstats import r_quantile
    top_kernel, _ = _top_kernel(obj)
    T = []
    for k in range(len(obj.sigma)):
        v = np.array([r["T_stat"][index, niche] if r["T_stat"] is not None else np.nan for r in obj.niche_DE[k]])
        T.append(np.clip(v, -abs(truncation_value), abs(truncation_value)))
    L_genes = list(obj.ref_expr.columns); Lv = _dense(obj.ref_expr)
    CT_filter = np.array([r_quantile(Lv[i], 0.25) for i in range(Lv.shape[0])])
    cand = set(np.asarray(L_genes)[Lv[niche] > CT_filter[niche]])
    lt = ltm.loc[:, [c for c in ltm.columns if c in cand]]
    genes_all = np.asarray(obj.gene_names)
    act = {}
    for lig in lt.columns:
        sig = T[int(top_kernel[L_genes.index(lig)])]
        keep = ~np.isnan(sig)
        gk = genes_all[keep]; sk = sig[keep]
        lv = lt.loc[lt.index.isin(set(gk))]
        m = np.isin(gk, lv.index.to_numpy()); gk, sk = gk[m], sk[m]
        col = lv.loc[gk, lig].to_numpy(float)
        col = (col - col.mean()) / col.std(ddof=1)
        top = _order_decreasing(col)[:K]
        w = col[top] / col[top].mean()
        act[lig] = float(np.sum(sk[top] * w) / np.sqrt(np.sum(w ** 2)))
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissues"); ap.add_argument("out")
    ap.add_argument("--prior", default="noisy,oracle"); ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--scenario", default="mixed")            # the exported tissues' name prefix (e.g. mixed_noniche)
    a = ap.parse_args()
    import pynichede as nde
    from pynichede.downstream import niche_LR_cell, _ligand_scores
    out = a.out.replace(".jsonl", f"_{a.worker}.jsonl") if a.n_workers > 1 else a.out
    done = {json.loads(l)["tissue"] for l in open(out)} if os.path.exists(out) else set()
    files = sorted(f for f in glob.glob(os.path.join(a.tissues, f"{a.scenario}_*.h5ad"))
                   if re.fullmatch(re.escape(a.scenario) + r"_\d+_(planted|null)", os.path.basename(f)[:-5]))
    for k, f in enumerate(files):
        if k % a.n_workers != a.worker:
            continue
        name = os.path.basename(f)[:-5]
        if name in done:
            continue
        t0 = time.time()
        seed = int(name[len(a.scenario) + 1:].split("_")[0])
        truth = json.load(open(os.path.join(a.tissues, f"{a.scenario}_{seed}_planted.truth.json")))
        ch = json.load(open(os.path.join(a.tissues, f"{a.scenario}_{seed}.channels.json")))
        adata = sc.read_h5ad(f)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        counts = pd.DataFrame(np.rint(X).astype(int), index=adata.obs_names, columns=adata.var_names)
        ct = adata.obs["cell_type"].astype(str).values
        types = sorted(set(ct))
        coord = pd.DataFrame(np.asarray(adata.obsm["spatial"], float), index=adata.obs_names, columns=["x", "y"])
        lib = nde.create_library_matrix(counts, pd.DataFrame({"cell": adata.obs_names, "cell_type": ct}), random_state=0)
        deconv = pd.DataFrame(np.array([[1.0 if c == t_ else 0.0 for t_ in types] for c in ct]), index=adata.obs_names, columns=types)
        obj = nde.create_nichede_object(counts, coord, lib.loc[types], deconv, sigma=[1.0, 100.0, 250.0])
        obj = nde.calculate_effective_niche(obj, cutoff=0.05)
        obj = nde.niche_DE(obj, num_cores=a.cores, C=150, M=10, gamma=0.8, Int=True, batch=True, self_EN=False, verbose=False)
        cts = [str(c) for c in obj.cell_types]
        shipped_nc = obj.num_cells
        onehot_nc = deconv.loc[obj.num_cells.index, obj.num_cells.columns].astype(float)
        db = build_db(ch)
        lr_mat = pd.DataFrame({"ligand": [f"g{g}" for g in db["lig"]], "receptor": [f"g{g}" for g in db["rec"]]})
        rec = dict(tissue=name, seed=seed, types=cts, db=db.to_dict("list"), res={})
        for prior in a.prior.split(","):
            ltm = build_prior(db, truth, X.shape[1], seed, prior)
            res = {}
            for ia, ta in enumerate(cts):
                for ib, tb in enumerate(cts):
                    if ia == ib:
                        continue
                    act = ligand_activity(obj, ltm, ib, ia)
                    try:
                        top = _ligand_scores(obj, ltm, ib, ia, 25, 50, 3)[0]
                    except Exception:
                        top = []
                    srt = sorted(act.values(), reverse=True)
                    cut = max(1.64, srt[49]) if len(srt) >= 50 else 1.64
                    assert set(top) == {l for l, v in act.items() if v >= cut}, (name, prior, ta, tb, top, act)
                    calls = {}
                    for lab, nc in (("lr", onehot_nc), ("lr_shipped", shipped_nc)):
                        obj.num_cells = nc
                        try:
                            lr = niche_LR_cell(obj, ta, tb, ltm, lr_mat)
                            calls[lab] = [[r.ligand, r.receptor] for r in lr.itertuples()]
                        except ValueError:
                            calls[lab] = []
                    obj.num_cells = shipped_nc
                    res[f"{int(ta[4:])}|{int(tb[4:])}"] = dict(**calls, activity={l: round(v, 4) for l, v in act.items()})
            rec["res"][prior] = res
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_called = {p: (sum(bool(v["lr"]) for v in r.values()), sum(bool(v["lr_shipped"]) for v in r.values()))
                    for p, r in rec["res"].items()}
        print(name, f"{time.time() - t0:.0f}s", "pairs called", n_called, flush=True)


if __name__ == "__main__":
    main()
