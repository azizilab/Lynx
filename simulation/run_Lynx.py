"""Fit LYNX on the simulated tissues and run its read-outs: the leak fit, C x C, C x C x G, C x C x P (the planted programs plus
decoys as the user-given P), and on the planted niche fields K x C, K x C x G, K x C x P.

    python run_Lynx.py <outdir> [--scenario mixed] [--seeds 250-259] [--aux default|noise] [--epochs 300] [--B 500]
                       [--worker K --n-workers N] [--from-readouts DIR] [--alpha A] [--no-niche]
--from-readouts: no fit; `model.evaluate`'s stored output of an earlier run (<DIR>/<family>/<tissue>.readouts.pkl, its obsm) is
reused and only the tests are run (the tests read the prior niche latent, the counts, positions and types). --alpha sets the tests'
exposure kernel decay (the fit's edge prior keeps the dataset's alpha); --no-niche skips the niche tests.

Per tissue (planted and its null twin): <outdir>/<family>/<tissue>.pt (state, NB dispersion, config, seed) and
<tissue>.readouts.pkl (the tables benchmarks/simulation/score.py reads); skip-if-exists.
"""
import os, sys, json, time, zlib, random, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, f"{ROOT}/models", f"{ROOT}/util", HERE]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import dataset, configs, vgae
import test_assoc
from simulate import make_tissue, niche_fields, programs_for

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graphs(t, n_subgraphs):
    return dataset.HeteroDataset(adatas_ref=t.adata_ref, adatas_query=t.adata_aux, n_subgraphs=n_subgraphs, r=t.radius,
                                 is_weighted=True, cluster_key="cell_type", alpha=0.5, verbose=False)


def fit(t, epochs=300, lr=1e-2, **cfg):
    """The revision's training: 4 partitions, no held-out partition, training seed = tissue seed, fixed epochs."""
    gd, seed = graphs(t, 4), t.seed
    pyro.clear_param_store(); pyro.set_rng_seed(seed); torch.manual_seed(seed)
    random.seed(zlib.crc32(json.dumps(dict(cfg, version="revised"), sort_keys=True).encode()) ^ seed)  # PyG module names
    m = vgae.HeteroAttnVGAE(configs.set_model_configs(graph_data=gd, act=nn.SiLU(), **cfg), device=DEV)
    svi, sched, _ = m.setup(m, configs.set_train_configs(n_epochs=epochs, lr=lr, device=DEV))
    parts = [gd[i].to(DEV) for i in range(len(gd))]
    for _ in range(epochs):
        m.train()
        for data in parts:
            svi.step(data)
        sched.step()
    return m.eval()


def load(t, path):
    """A fit saved by this script on tissue t."""
    blob = torch.load(path, map_location=DEV)
    cfg = {k: v for k, v in blob["cfg"].items() if k not in ("kappa_lambda_start", "kappa_anneal_frac", "version")}
    pyro.clear_param_store()
    m = vgae.HeteroAttnVGAE(configs.set_model_configs(graph_data=graphs(t, 4), act=nn.SiLU(), **{"alpha": 0.5, **cfg}), device=DEV).to(DEV)
    m.load_state_dict(blob["state"])
    pyro.param("theta", blob["theta"].to(DEV), constraint=pyro.distributions.constraints.positive)
    return m.eval()


def readouts(m, t, B=500, stored=None, alpha=None, niche=True):
    """Everything the benchmark scores from one fit: `test_assoc.get_test_interactions` on `model.evaluate`'s output (or, with
    `stored`, the obsm of an earlier run's read-outs), with the planted niche fields as the niche and the planted programs plus
    decoys as the program set."""
    ad, aux = t.adata_ref.copy(), t.adata_aux.copy()
    if stored is None:
        m.evaluate(ad, aux, graph_data=graphs(t, 1), device=DEV)
    else:
        for k, v in stored["obsm"].items():
            ad.obsm[k] = np.asarray(v)
        ad.uns["lynx"] = dict(cluster_key="cell_type", radius=float(t.radius), alpha=0.5)
    if alpha is not None:                                                                # the tests' exposure kernel (1 + d)^-alpha; the fit's prior is untouched
        ad.uns["lynx"]["alpha"] = float(alpha)
    planted = make_tissue(t.scenario, t.seed, False, aux=t.aux) if t.null else t      # the null twin is scored against its planted twin's niche and programs
    out = test_assoc.get_test_interactions(ad, fields=niche_fields(planted) if niche else None, programs=programs_for(planted), seed=t.seed, B=B)
    out.update(obsm={k: np.asarray(ad.obsm[k]) for k in ("X_z_prior", "X_z", "X_kappa", "X_msg", "X_s_hat")},
               names=list(ad.obs["cell_type"].cat.categories), genes=list(ad.var_names), programs=programs_for(planted))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("--scenario", default="mixed"); ap.add_argument("--seeds", default="250-259")
    ap.add_argument("--aux", default="default"); ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1); ap.add_argument("--from-readouts", default=None)
    ap.add_argument("--alpha", type=float, default=None); ap.add_argument("--no-niche", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
    fam = "lynx_revised" + ("_noiseaux" if a.aux == "noise" else "")
    os.makedirs(os.path.join(a.outdir, fam), exist_ok=True)
    jobs = [(s, null) for s in range(lo, hi + 1) for null in (False, True)]
    for k, (seed, null) in enumerate(jobs):
        name = f"{a.scenario}_{seed}_{'null' if null else 'planted'}"
        f_fit, f_ro = (os.path.join(a.outdir, fam, name + ext) for ext in (".pt", ".readouts.pkl"))
        if k % a.n_workers != a.worker or os.path.exists(f_ro):
            continue
        t0 = time.time()
        t = make_tissue(a.scenario, seed, null, aux=a.aux)
        if a.from_readouts:
            stored = pd.read_pickle(os.path.join(a.from_readouts, fam, name + ".readouts.pkl"))
            pd.to_pickle(readouts(None, t, a.B, stored=stored, alpha=a.alpha, niche=not a.no_niche), f_ro)
            print(fam, name, f"{time.time() - t0:.0f}s (tests on stored evaluate output)", flush=True)
            continue
        if os.path.exists(f_fit):
            m = load(t, f_fit)
        else:
            m = fit(t, a.epochs)
            torch.save(dict(state=m.state_dict(), theta=pyro.param("theta").detach().cpu(), cfg={}, seed=seed), f_fit)
        pd.to_pickle(readouts(m, t, a.B, alpha=a.alpha, niche=not a.no_niche), f_ro)
        print(fam, name, f"{time.time() - t0:.0f}s", getattr(m, "kappa_solve_stats", ""), flush=True)


if __name__ == "__main__":
    main()
