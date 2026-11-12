"""AMICI (Hong et al. 2025; github.com/azizilab/amici) tuned per tissue as its guide prescribes (docs/examples/
basic_usage.ipynb; the sweep grid of xenium/xenium_sweep.yaml), on the exported simulated tissues (export_tissues.py).
Runs in the amici environment (~/future/envs/amici, python -I):

    python run_AMICI.py <tissue dir> <out jsonl> [--attention-dir DIR] [--worker K --n-workers N] [--seed 18]
"""
import os, glob, json, time, argparse, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc

GRID = dict(end_attention_penalty=[1e-3, 1e-2], schedule=[(5, 15), (10, 20)], value_l1_penalty_coef=[1e-5, 1e-3])


def test_region(xy, ct, frac=0.10, n_grid=15):
    from scipy.spatial.distance import jensenshannon
    T = int(ct.max()) + 1
    lo, hi = xy.min(0), xy.max(0)
    side = np.sqrt(frac * np.prod(hi - lo))
    full = np.bincount(ct, minlength=T) / len(ct)
    best = None
    for cx in np.linspace(lo[0] + side / 2, hi[0] - side / 2, n_grid):
        for cy in np.linspace(lo[1] + side / 2, hi[1] - side / 2, n_grid):
            m = (np.abs(xy[:, 0] - cx) <= side / 2) & (np.abs(xy[:, 1] - cy) <= side / 2)
            if not (0.08 * len(ct) <= m.sum() <= 0.12 * len(ct)):
                continue
            jsd = float(jensenshannon(np.bincount(ct[m], minlength=T) / m.sum(), full, base=2.0) ** 2)
            if best is None or jsd < best[0]:
                best = (jsd, m, (float(cx), float(cy), float(side)))
    return best


def choose_k(xy, radius=50.0, candidates=(30, 40, 50, 60)):
    from sklearn.neighbors import NearestNeighbors
    d = NearestNeighbors(n_neighbors=max(candidates) + 1).fit(xy).kneighbors(xy)[0][:, 1:]
    med = {int(k): float(np.median(d[:, k - 1])) for k in candidates}
    ok = [k for k in candidates if med[k] >= 1.2 * radius]
    return (min(ok) if ok else 50), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissues"); ap.add_argument("out")
    ap.add_argument("--attention-dir", default=None)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--max-epochs", type=int, default=400)
    a = ap.parse_args()
    import torch
    import pytorch_lightning as pl
    from amici import AMICI
    from amici.callbacks import AttentionPenaltyMonitor
    out = a.out.replace(".jsonl", f"_{a.worker}.jsonl") if a.n_workers > 1 else a.out
    done = {json.loads(l)["tissue"] for l in open(out)} if os.path.exists(out) else set()
    files = sorted(glob.glob(os.path.join(a.tissues, "*.h5ad")))
    for kf, f in enumerate(files):
        if kf % a.n_workers != a.worker:
            continue
        name = os.path.basename(f)[:-5]
        if name in done:
            continue
        t0 = time.time()
        adata = sc.read_h5ad(f)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata); sc.pp.log1p(adata)
        xy = np.asarray(adata.obsm["spatial"], float)
        types = sorted(adata.obs["cell_type"].cat.categories)
        ct = np.array([types.index(c) for c in adata.obs["cell_type"].astype(str)])
        jsd, test, box = test_region(xy, ct)
        adata.obs["in_test"] = test
        k_nn, k_med = choose_k(xy)
        train = adata[~test].copy()
        grid = []
        best = None
        for end, (es, ee), l1 in itertools.product(GRID["end_attention_penalty"], GRID["schedule"], GRID["value_l1_penalty_coef"]):
            pl.seed_everything(a.seed)
            penalty = dict(start_val=1e-6, end_val=end, epoch_start=es, epoch_end=ee, flavor="linear")
            params = dict(n_heads=8, n_query_dim=128, n_head_size=32, n_nn_embed=256, n_nn_embed_hidden=512,
                          attention_dummy_score=3.0, neighbor_dropout=0.1, attention_penalty_coef=penalty["start_val"],
                          value_l1_penalty_coef=l1)
            tr_ = train.copy()
            AMICI.setup_anndata(tr_, labels_key="cell_type", coord_obsm_key="spatial", n_neighbors=k_nn)
            model = AMICI(tr_, **params)
            model.train(max_epochs=a.max_epochs, batch_size=512, plan_kwargs=dict(lr=1e-3), early_stopping=True,
                        early_stopping_monitor="elbo_validation", early_stopping_patience=20, check_val_every_n_epoch=1,
                        callbacks=[AttentionPenaltyMonitor(**penalty)])
            full = adata.copy()
            AMICI.setup_anndata(full, labels_key="cell_type", coord_obsm_key="spatial", n_neighbors=k_nn)
            idx = np.where(test)[0]
            elbo = float(model.get_elbo(full, indices=idx, batch_size=128))
            try:
                rec_loss = float(np.mean(np.asarray(model.get_reconstruction_error(full, indices=idx, batch_size=128)["reconstruction_loss"])))
            except Exception:
                rec_loss = float("nan")
            ep = int(model.history["elbo_validation"].shape[0]) if "elbo_validation" in getattr(model, "history", {}) else None
            cfg = dict(end_attention_penalty=end, epoch_start=es, epoch_end=ee, value_l1_penalty_coef=l1)
            grid.append(dict(cfg, test_elbo=elbo, test_reconstruction_loss=rec_loss, epochs=ep))
            print(name, cfg, f"test ELBO {elbo:.3f}", f"rec {rec_loss:.3f}", f"epochs {ep}", flush=True)
            if best is None or elbo > best[0]:
                best = (elbo, model, cfg, ep)
            else:
                del model
            torch.cuda.empty_cache()
        _, model, cfg, ep = best
        AMICI.setup_anndata(adata, labels_key="cell_type", coord_obsm_key="spatial", n_neighbors=k_nn)
        if a.attention_dir:
            os.makedirs(a.attention_dir, exist_ok=True)
            att = model.get_attention_patterns(adata=adata)._attention_patterns_df
            nn_idx = np.asarray(adata.obsm["_nn_idx"])
            A = att[[f"neighbor_{j}" for j in range(nn_idx.shape[1])]].astype(float).to_numpy()
            ids = att["cell_idx"].astype(int).to_numpy()
            S = np.zeros(nn_idx.shape); C = np.zeros(adata.n_obs)
            np.add.at(S, ids, A); np.add.at(C, ids, 1.0)
            np.savez_compressed(os.path.join(a.attention_dir, f"{name}.attn.npz"), nn_idx=nn_idx.astype(np.int32),
                                attn=(S / np.maximum(C, 1)[:, None]).astype(np.float32))
        abl = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
        df = abl._ablation_scores_df
        W = abl._get_interaction_weight_matrix()
        Wv = W.loc[types, types].values
        rec = dict(tissue=name, epochs_run=ep, types=types, genes=list(adata.var_names), weight=Wv.tolist(), receivers={},
                   tuning=dict(test_region_jsd=jsd, test_frac=float(test.mean()), test_box=box, n_neighbors=k_nn,
                               kth_neighbour_median_um=k_med, grid=grid, chosen=cfg,
                               weight_threshold_q86=float(np.quantile(Wv.ravel(), 0.86))))
        for b in types:
            sub = df[df["cell_type"] == b]
            genes = sub["gene"].astype(str).tolist() if "gene" in sub.columns else list(adata.var_names)
            e = dict(genes=genes)
            for a_ in types:
                if a_ == b or f"{a_}_diff" not in sub.columns:
                    continue
                e[a_] = dict(diff=np.round(sub[f"{a_}_diff"].astype(float).values, 5).tolist(),
                             nl10_padj=np.round(sub[f"{a_}_nl10_pval_adj"].astype(float).values, 4).tolist() if f"{a_}_nl10_pval_adj" in sub.columns else None,
                             z=np.round(sub[f"{a_}_z_value"].astype(float).values, 4).tolist() if f"{a_}_z_value" in sub.columns else None)
            rec["receivers"][b] = e
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, "chosen", cfg, f"{time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
