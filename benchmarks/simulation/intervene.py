"""Disentanglement by intervention. Each planted tissue is measured four more times, each with its own count noise:
reference, niche intervened (new niche fields and environment field, so new auxiliary data), intrinsic intervened (new
within-type states) and measured again (floor). The model fitted on the original tissue reads each version; for every
latent (kappa - mu, the posterior niche latent z; SIMVI's intrinsic / spatial latents), centred within cell types:
    r(A, B) = <A, B> / (|A| |B|);  replaced(V) = 1 - (r(reference, V) / r(reference, floor))^2, clipped to [0, 1]
(Spearman's attenuation correction: the share of the latent's reproducible variance V replaces); specificity =
own / (own + other). Median over tissues with a 95% bootstrap interval, and the per-tissue values (the report draws mean +- SE).

    python intervene.py <fits dir> <out dir> [--scenario mixed] [--simvi]     (fits: run_Lynx.py's family folders)
    -> <out dir>/<family>_<seed>.npz, versions/<tissue>_<version>.h5ad (for intervene_SIMVI.py), INTERVENE.md / .json
"""
import os, sys, glob, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [f"{ROOT}/simulation", HERE]
import run_Lynx as R  # noqa: E402
from simulate import make_tissue  # noqa: E402

SEEDS, VERS, NBOOT, MIN_REL = range(250, 260), ("reference", "niche", "intrinsic", "floor"), 2000, 0.05
FAMILIES = {"LYNX recommended": ("lynx_revised", "default"), "LYNX recommended, noise auxiliary": ("lynx_revised_noiseaux", "noise")}


def versions(seed):
    return dict(reference={"counts": 920_000 + seed}, niche={"counts": 921_000 + seed, "niche": 900_000 + seed},
                intrinsic={"counts": 922_000 + seed, "intrinsic": 910_000 + seed}, floor={"counts": 923_000 + seed})


def read_versions(fits, out, scen):
    for seed in SEEDS:
        for m, (fam, aux) in FAMILIES.items():
            o = os.path.join(out, f"{fam}_{seed}.npz")
            f = glob.glob(os.path.join(fits, fam, f"{scen}_{seed}_planted*.pt"))
            if os.path.exists(o) or not f:
                continue
            arrs, model = {}, None
            for v, iv in versions(seed).items():
                t = make_tissue(scen, seed, False, aux=aux, intervene=iv)
                model = model or R.load(t, f[0])
                ad, ax = t.adata_ref.copy(), t.adata_aux.copy()
                model.evaluate(ad, ax, graph_data=R.graphs(t, 1), device=R.DEV)
                arrs[f"kappa_{v}"] = ad.obsm["X_kappa"] - ad.uns["lynx"]["kappa_mu"][t.ct]
                arrs[f"z_{v}"], arrs[f"zprior_{v}"] = ad.obsm["X_z"], ad.obsm["X_z_prior"]
                arrs[f"true_intrinsic_{v}"] = t.truth["kappa"] - t.truth["mu"][t.ct]
                arrs[f"true_niche_{v}"] = np.column_stack([t.truth["z_niche"], t.truth["env"]])
                if aux == "default":
                    os.makedirs(os.path.join(out, "versions"), exist_ok=True)
                    h = os.path.join(out, "versions", f"{scen}_{seed}_{v}.h5ad")
                    if not os.path.exists(h):
                        import anndata as anndata
                        a = anndata.AnnData(np.asarray(t.adata_ref.X, np.float32))
                        a.obs["cell_type"] = [f"type{c}" for c in t.ct]
                        a.obs["cell_type"] = a.obs["cell_type"].astype("category")
                        a.obsm["spatial"], a.var_names = np.asarray(t.xy, float), [f"g{g}" for g in range(t.G)]
                        a.write_h5ad(h)
            np.savez_compressed(o, ct=t.ct.astype(np.int16), **{k: np.asarray(v_, np.float32) for k, v_ in arrs.items()})
            print(fam, seed, flush=True)


def vcorr(Aa, Bb, ct):
    """Correlation of the type-centred versions over cells and dimensions."""
    Ac, Bc = np.array(Aa, float), np.array(Bb, float)
    for c in np.unique(ct):
        Ac[ct == c] -= Ac[ct == c].mean(0)
        Bc[ct == c] -= Bc[ct == c].mean(0)
    den = np.sqrt((Ac ** 2).sum() * (Bc ** 2).sum())
    return float((Ac * Bc).sum() / den) if den > 0 else np.nan


def score(out, simvi=False):
    lat = {"kappa": ("kappa", "intrinsic"), "z": ("z", "niche"), "zprior": ("zprior", "niche")}
    methods = {m: (fam, lat) for m, (fam, _) in FAMILIES.items()}
    if simvi:
        methods["SIMVI"] = ("simvi", {"kappa": ("intrinsic", "intrinsic"), "z": ("spatial", "niche")})
    other, rng, res = {"intrinsic": "niche", "niche": "intrinsic"}, np.random.default_rng(0), {}
    for m, (pre, lats) in methods.items():
        rows = []
        for seed in SEEDS:
            f = os.path.join(out, f"{pre}_{seed}.npz")
            if not os.path.exists(f):
                continue
            z, r = np.load(f), {}
            ct = z["ct"].astype(int)
            for lab, (arr, own) in lats.items():
                V = {v: z[f"{arr}_{v}"] for v in VERS}
                ok = np.all([np.isfinite(x).all(1) for x in V.values()], axis=0)
                rel = vcorr(V["reference"][ok], V["floor"][ok], ct[ok])
                rep = {v: (1.0 - float(np.clip(vcorr(V["reference"][ok], V[v][ok], ct[ok]) / rel, 0.0, 1.0)) ** 2
                           if np.isfinite(rel) and rel >= MIN_REL else np.nan) for v in ("niche", "intrinsic")}
                r.update({f"{lab}_reliability": rel, f"{lab}_own": rep[own], f"{lab}_cross": rep[other[own]]})
                o_, x_ = rep[own], rep[other[own]]
                r[f"{lab}_spec"] = o_ / (o_ + x_) if np.isfinite(o_) and np.isfinite(x_) and o_ + x_ > 1e-9 else np.nan
            rows.append(r)
        if rows:
            res[m] = {"per_tissue": rows}
            for k in rows[0]:
                v = np.array([x[k] for x in rows if np.isfinite(x[k])])
                b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(NBOOT)] if len(v) else [np.nan]
                res[m][k] = dict(value=float(np.median(v)) if len(v) else np.nan, ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    json.dump(res, open(os.path.join(out, "INTERVENE.json"), "w"), indent=1)
    cols = [("kappa_own", "κ: replaced by the intrinsic intervention"), ("kappa_cross", "κ: by the niche intervention (↓)"), ("kappa_spec", "κ specificity"),
            ("z_own", "z (posterior): replaced by the niche intervention"), ("z_cross", "z: by the intrinsic intervention (↓)"), ("z_spec", "z specificity")]
    f = lambda e: "–" if e is None or not np.isfinite(e["value"]) else f"{e['value']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]"
    L = ["| method | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {m} | " + " | ".join(f(d.get(k)) for k, _ in cols) + " |" for m, d in res.items()]
    open(os.path.join(out, "INTERVENE.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fits"); ap.add_argument("out"); ap.add_argument("--scenario", default="mixed"); ap.add_argument("--simvi", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    read_versions(a.fits, a.out, a.scenario)
    score(a.out, a.simvi)
