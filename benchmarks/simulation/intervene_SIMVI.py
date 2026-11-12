"""SIMVI in the intervention test: trained on the original tissue as run_SIMVI.py trains it (the paper's settings), then its
intrinsic and spatial latents on each version written by intervene.py. get_latent_representation reads the model's own
AnnData, so each version's counts are placed there (checked: the original counts reproduce the trained latents).

    ~/future/envs/simvi/bin/python intervene_SIMVI.py <intervene out dir> <tissue dir> [--scenario mixed] [seed ...]   -> simvi_<seed>.npz
"""
import os, time, argparse
import numpy as np
import anndata as ad
import torch
import scvi
from simvi.model.simvi import SimVIModel

_ap = argparse.ArgumentParser()
_ap.add_argument("out"); _ap.add_argument("tissues"); _ap.add_argument("seeds", nargs="*", type=int); _ap.add_argument("--scenario", default="mixed")
ARGS = _ap.parse_args()
OUT, TIS, SCEN = ARGS.out, ARGS.tissues, ARGS.scenario                   # intervene.py's out dir (its versions/), the exported tissues
H5 = os.path.join(OUT, "versions")
VERS = ("reference", "niche", "intrinsic", "floor")


def dense(A):
    return np.asarray(A.X.toarray() if hasattr(A.X, "toarray") else A.X, np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    for seed in (ARGS.seeds or range(250, 260)):
        out = os.path.join(OUT, f"simvi_{seed}.npz")
        if os.path.exists(out):
            continue
        files = [os.path.join(H5, f"{SCEN}_{seed}_{v}.h5ad") for v in VERS]
        t0 = time.time()
        A = ad.read_h5ad(os.path.join(TIS, f"{SCEN}_{seed}_planted.h5ad")); A.X = dense(A)
        A.obs["cell_type"] = A.obs["cell_type"].astype(str).astype("category")
        ct = np.array([int(c.replace("type", "")) for c in A.obs["cell_type"]])
        scvi.settings.seed = seed; torch.manual_seed(seed); np.random.seed(seed)
        SimVIModel.setup_anndata(A)
        ei = SimVIModel.extract_edge_index(A, spatial_key="spatial", n_neighbors=10)
        model = SimVIModel(A)
        model.train(ei, max_epochs=100, batch_size=500, mae_epochs=25, use_gpu=True)
        ei = ei.to(next(model.module.parameters()).device)
        lat = lambda kind: model.get_latent_representation(ei, representation_kind=kind)
        X0 = np.array(model.adata.X, copy=True)
        arrs = dict(ct=ct.astype(np.int16), intrinsic_trained=lat("intrinsic"), spatial_trained=lat("interaction"))
        for v, f in zip(VERS, files):
            B = ad.read_h5ad(f)
            assert np.allclose(B.obsm["spatial"], A.obsm["spatial"]) and list(B.obs["cell_type"].astype(str)) == list(A.obs["cell_type"].astype(str))
            model.adata.X = dense(B)
            arrs[f"intrinsic_{v}"], arrs[f"spatial_{v}"] = lat("intrinsic"), lat("interaction")
        model.adata.X = X0
        assert not np.allclose(arrs["intrinsic_reference"], arrs["intrinsic_trained"], atol=1e-3), "counts swap does not reach the model"
        assert np.allclose(lat("intrinsic"), arrs["intrinsic_trained"], atol=1e-5), "original counts do not reproduce the trained latents"
        np.savez_compressed(out, **arrs)
        print(seed, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
