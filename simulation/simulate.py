r"""Simulated tissues for the cell-cell interaction benchmark.

4,000 cells of 6 types, 256 genes. Each true pair (sender a -> receiver b) is carried by a ligand of a and a receptor of b:
a receiver's dose is the ligand-weighted distance kernel exp(-d / lam) over its senders within the 50 um graph radius,
saturating at its median, times a receptor gate; the dose moves the pair's own 8-gene receiver program, with a gain that
depends on the gradient t (constant, low-niche band, rising, fading). Scenarios (SCENARIOS): `mixed` (the default) plants
two pairs at contact range (lam 8 um) and two near-flat inside the neighbourhood (lam 80 um); `mixed_swap` exchanges the
two; `mixed_noniche` is the second simulation (no niche, no environment, constant gains); `unified` and `unified_noniche`
are the single-range originals (every pair lam 20 um). Three decoy channels per pair; sender sub-state programs (one carries the true ligand, one follows the
environment); decoy types placed next to senders. Niche: three smooth fields shift every cell's state, a hypoxia-like
field (measured in the auxiliary modality) switches on its own module, enriches types 0 and 5 and loads on half of every
pair's response genes. 25% of every cell's counts are assigned to its 6 nearest cells (segmentation bleed).
`intervene` redraws one source ({'niche' | 'intrinsic' | 'counts': seed}) with every other draw kept (the
disentanglement test); `aux='noise'` replaces the auxiliary modality by noise (ablation).
"""
import copy, json, os
from dataclasses import dataclass, field as dc_field
from types import SimpleNamespace
import numpy as np
import anndata as ad
from scipy.spatial import cKDTree

EPS = 1e-12


class Field:
    """Smooth random field (random Fourier features), standardised on the tissue."""

    def __init__(self, rng, scale, size, m=64):
        self.w = rng.normal(0, 1.0 / scale, size=(m, 2))
        self.b = rng.uniform(0, 2 * np.pi, size=m)
        self.a = rng.normal(size=m)
        raw = self._raw(np.stack(np.meshgrid(np.linspace(0, size, 50), np.linspace(0, size, 50)), -1).reshape(-1, 2))
        self.mu, self.sd = raw.mean(), raw.std()

    def _raw(self, xy):
        return np.cos(xy @ self.w.T + self.b) @ self.a

    def __call__(self, xy):
        return (self._raw(xy) - self.mu) / self.sd


@dataclass
class Config:
    n_cells: int = 4000
    size: float = 1000.0
    n_types: int = 6
    patch_scale: float = 200.0
    decoys: list = dc_field(default_factory=lambda: [(4, 0), (5, 3)])   # (decoy type, placed next to type)
    D: int = 8
    n_niche: int = 3                    # niche fields; the first is the gradient
    n_genes: int = 256
    module_size: int = 10
    field_scale: float = 300.0
    type_sep: float = 1.5
    kappa_sd: float = 0.7
    niche_effect: float = 0.6
    graph_radius: float = 50.0
    lam: float = 20.0
    r_bigraph: float = 30.0
    pixel: float = 20.0
    aux_dim: int = 8
    aux_noise: float = 0.3
    pairs: list = dc_field(default_factory=list)
    null: bool = False                  # no response (same tissue, same colocalization)
    lr_decoys: int = 3
    lr_strength: float = 2.0
    env_genes: int = 10
    env_effect: float = 1.5
    env_comp: float = 0.5
    env_types: list = dc_field(default_factory=lambda: [0, 5])
    env_scale: float = 250.0
    env_resp_frac: float = 0.5
    env_resp_effect: float = 1.5
    bleed: float = 0.25
    bleed_k: int = 6
    type_genes: int = 6
    type_gene_level: float = 2.5
    prog_genes: int = 6
    prog_scale: float = 120.0
    prog_sd_cell: float = 0.5
    prog_level: float = 2.0
    prog_load: float = 0.8
    prog_env_corr: float = 0.6
    lig_own_sd: float = 0.5
    resp_genes: int = 8
    resp_level: float = 2.0
    resp_noise: float = 0.4
    resp_effect: float = 2.0
    loading: float = 0.8
    counts_per_cell: float = 300.0
    dispersion: float = 5.0
    lib_sd: float = 0.2


PAIRS = [dict(sender=0, receiver=1, kind="const", beta0=2.0),                 # strong, everywhere
         dict(sender=2, receiver=1, kind="band", beta0=1.5, lo=-99.0, hi=0.0),  # low niche only
         dict(sender=3, receiver=2, kind="relu", beta0=1.0, beta1=-0.5),        # rising along the gradient
         dict(sender=1, receiver=3, kind="linear", beta0=0.6, beta1=-0.5)]      # fading along the gradient
# no niche effect: the fields are measured but act on nothing; the gradient pairs at their mean gain under t ~ N(0, 1)
NONICHE = dict(niche_effect=0.0, env_effect=0.0, env_resp_frac=0.0, env_resp_effect=0.0, env_comp=0.0, prog_env_corr=0.0)
PAIRS_NONICHE = [dict(PAIRS[0]), dict(sender=2, receiver=1, kind="const", beta0=0.75),
                 dict(sender=3, receiver=2, kind="const", beta0=0.70), dict(sender=1, receiver=3, kind="const", beta0=0.63)]
# planted ranges inside the 50 um neighbourhood: contact range (lam 8 um: half weight at 5.5 um, 0.2% at the radius) and
# near-flat (lam 80 um: 54% at the radius); MIX assigns them to the four pairs, the swap gives every pair the other range, so
# range is not confounded with a pair's strength or niche dependence
RANGES = dict(short=8.0, long=80.0)
MIX = ("short", "long", "long", "short")
ranged = lambda pairs, kinds: [dict(p, lam=RANGES[k]) for p, k in zip(pairs, kinds)]
SCENARIOS = {
    "mixed": dict(pairs=ranged(PAIRS, MIX)),                                                        # the default
    "mixed_swap": dict(pairs=ranged(PAIRS, [{"short": "long", "long": "short"}[k] for k in MIX])),
    "mixed_noniche": dict(NONICHE, pairs=ranged(PAIRS_NONICHE, MIX)),                              # the second simulation
    "unified": dict(pairs=PAIRS),                                                                   # single range: every pair lam 20 um
    "unified_noniche": dict(NONICHE, pairs=PAIRS_NONICHE),
}


def lam_edges(cfg, ct_src, ct_dst):
    """Per-edge range of the planted influence kernel exp(-d / lam): a planted pair's own `lam` where it sets one, else cfg.lam."""
    L = np.full((cfg.n_types, cfg.n_types), float(cfg.lam))
    for p in cfg.pairs:
        L[p["sender"], p["receiver"]] = float(p.get("lam", cfg.lam))
    return L[ct_src, ct_dst]


def gain(pair, t):
    k = pair["kind"]
    if k == "const":
        return np.full_like(t, pair["beta0"])
    if k == "band":
        return np.where((t > pair["lo"]) & (t < pair["hi"]), pair["beta0"], 0.0)
    if k == "relu":
        return pair["beta0"] * np.maximum(0.0, t - pair.get("beta1", 0.0))
    return np.maximum(0.0, pair["beta0"] + pair["beta1"] * t)


def _logsoftmax(L):
    m = L.max(1, keepdims=True)
    return L - m - np.log(np.exp(L - m).sum(1, keepdims=True))


def simulate(cfg: Config, seed: int = 0, intervene: dict = None):
    """-> (adata_ref, adata_aux, truth)."""
    iv = intervene or {}
    rng = np.random.default_rng(seed)
    N, T, D, G = cfg.n_cells, cfg.n_types, cfg.D, cfg.n_genes
    xy = rng.uniform(0, cfg.size, size=(N, 2))
    # ---- niche latent on the pixel grid, unpooled to cells as LYNX's dataset does
    gx, gy = np.meshgrid(np.arange(cfg.pixel / 2, cfg.size, cfg.pixel), np.arange(cfg.pixel / 2, cfg.size, cfg.pixel))
    qxy = np.c_[gx.ravel(), gy.ravel()]
    z_pix = np.stack([Field(rng, cfg.field_scale, cfg.size)(qxy) for _ in range(cfg.n_niche)], 1)
    near = cKDTree(qxy).query_ball_point(xy, r=cfg.r_bigraph)
    pool = lambda Zp: np.stack([Zp[p].mean(0) if len(p) else np.zeros(Zp.shape[1]) for p in near])
    z_niche = pool(z_pix)
    stdz = lambda v: (v - v.mean()) / v.std()
    t = stdz(z_niche[:, 0])
    if iv.get("niche") is not None:                        # placement keeps the original niche (types kept)
        nrng = np.random.default_rng(iv["niche"])
        z_pix = np.stack([Field(nrng, cfg.field_scale, cfg.size)(qxy) for _ in range(cfg.n_niche)], 1)
        z_niche = pool(z_pix)
        t = stdz(z_niche[:, 0])
    A = rng.normal(size=(cfg.n_niche, cfg.aux_dim)) / np.sqrt(cfg.n_niche)
    U = (z_pix @ A + rng.normal(0, cfg.aux_noise, size=(len(qxy), cfg.aux_dim))).astype(np.float32)
    # ---- environment field (separate stream), measured in the aux modality
    erng = np.random.default_rng(seed + 3_333)
    h_pix = Field(erng, cfg.env_scale, cfg.size)(qxy)
    h_cell = np.stack([h_pix[p].mean() if len(p) else 0.0 for p in near])
    mu_h, sd_h = h_cell.mean(), h_cell.std()
    env_z = env_z_place = (h_cell - mu_h) / sd_h
    if iv.get("niche") is not None:                        # the environment field is part of the niche
        h_pix = Field(np.random.default_rng(iv["niche"] + 3_333), cfg.env_scale, cfg.size)(qxy)
        h_cell = np.stack([h_pix[p].mean() if len(p) else 0.0 for p in near])
        mu_h, sd_h = h_cell.mean(), h_cell.std()
        env_z = (h_cell - mu_h) / sd_h
    env = 1.0 / (1.0 + np.exp(-2.0 * env_z))
    U = np.c_[U, (h_pix - mu_h) / sd_h + erng.normal(0, cfg.aux_noise, len(qxy))].astype(np.float32)
    # ---- cell types: patches, decoys next to their partner type, env_types enriched where the field is high
    centers = rng.uniform(0.15 * cfg.size, 0.85 * cfg.size, size=(T, 2))
    for dec, partner in cfg.decoys:
        ang = rng.uniform(0, 2 * np.pi)
        centers[dec] = centers[partner] + 0.5 * cfg.patch_scale * np.array([np.cos(ang), np.sin(ang)])
    p = np.exp(-((xy[:, None] - centers[None]) ** 2).sum(-1) / (2 * cfg.patch_scale ** 2)) + 0.02
    if cfg.env_comp > 0:
        for a in cfg.env_types:
            p[:, a] *= np.exp(cfg.env_comp * env_z_place)
    p /= p.sum(1, keepdims=True)
    ct = (p.cumsum(1) < rng.uniform(size=(N, 1))).sum(1)
    # ---- latent states
    mu = rng.normal(0, cfg.type_sep, size=(T, D))
    eps_k = rng.normal(size=(N, D))
    if iv.get("intrinsic") is not None:
        eps_k = np.random.default_rng(iv["intrinsic"]).normal(size=(N, D))
    kappa = mu[ct] + cfg.kappa_sd * eps_k
    B = rng.normal(0, 1.0, size=(D, cfg.n_niche)) / np.sqrt(cfg.n_niche) * cfg.niche_effect
    niche = z_niche @ B.T
    # ---- colocalization: distance kernel over the graph neighbours of other types
    nb = cKDTree(xy).query_ball_point(xy, r=cfg.graph_radius)
    src = np.concatenate([[j for j in js if j != i] for i, js in enumerate(nb)]).astype(int)
    dst = np.concatenate([[i] * (len(js) - 1) for i, js in enumerate(nb)]).astype(int)
    k = np.exp(-np.linalg.norm(xy[src] - xy[dst], axis=1) / lam_edges(cfg, ct[src], ct[dst]))
    E = np.zeros((N, T))
    np.add.at(E, (dst, ct[src]), k * (ct[src] != ct[dst]))
    rng.normal(size=(T, D)), rng.normal(size=(T, T, D))       # (latent response programs, unused: kept for the stream)
    # ---- ligand-receptor channels (separate stream): per pair the true one and lr_decoys decoys
    channels, lr_level = [], {}
    lrng = np.random.default_rng(seed + 9_999)
    g0 = D * cfg.module_size + 2 * T
    for pi, pr in enumerate(cfg.pairs):
        a, b = pr["sender"], pr["receiver"]
        for m in range(1 + cfg.lr_decoys):
            gl, gr = g0 + 2 * (1 + cfg.lr_decoys) * pi + 2 * m, g0 + 2 * (1 + cfg.lr_decoys) * pi + 2 * m + 1
            L = np.where((ct == a) & (lrng.uniform(size=N) < 0.6), cfg.lr_strength * lrng.lognormal(0, 0.5, N), 0.0)
            R = np.where((ct == b) & (lrng.uniform(size=N) < 0.7), cfg.lr_strength * lrng.lognormal(0, 0.5, N), 0.0)
            lr_level[gl], lr_level[gr] = L, R
            channels.append(dict(sender=a, receiver=b, lig=gl, rec=gr, true=(m == 0)))
    # ---- programs (separate stream): identity genes, sender sub-states, receiver response programs
    prog, resp, ident, lig_dose, pair_dose = {}, {}, {}, {}, {}
    prng = np.random.default_rng(seed + 11_111)
    gp = g0 + 2 * (1 + cfg.lr_decoys) * len(cfg.pairs)
    for tt in range(T):
        ident[tt] = list(range(gp, gp + cfg.type_genes))
        gp += cfg.type_genes

    def activity(cells, follow=None, rho=0.0):
        f = Field(prng, cfg.prog_scale, cfg.size)(xy[cells])
        f = (f - f.mean()) / (f.std() + 1e-12)
        if follow is not None and rho > 0:
            f = np.sqrt(1 - rho ** 2) * f + rho * (follow - follow.mean()) / (follow.std() + 1e-12)
        pi = np.zeros(N)
        pi[cells] = f + cfg.prog_sd_cell * prng.normal(size=len(cells))
        return pi
    for a in sorted({pr["sender"] for pr in cfg.pairs}):
        ca = np.where(ct == a)[0]
        prog[a] = {}
        for kind in ("active", "inert"):
            pi = activity(ca, env_z[ca] if kind == "inert" else None, cfg.prog_env_corr)
            prog[a][kind] = dict(pi=pi, genes=list(range(gp, gp + cfg.prog_genes)), w=prng.uniform(0.6, 1.4, cfg.prog_genes))
            gp += cfg.prog_genes
    for pr in cfg.pairs:
        a, b = pr["sender"], pr["receiver"]
        v = prng.uniform(0.6, 1.4, cfg.resp_genes) * np.where(np.arange(cfg.resp_genes) < 6, 1.0, -1.0)
        resp[(a, b)] = dict(genes=list(range(gp, gp + cfg.resp_genes)), v=v, xi=np.where(ct == b, prng.normal(size=N), 0.0))
        gp += cfg.resp_genes
    assert gp <= G - cfg.env_genes, "not enough genes"
    for pr in cfg.pairs:            # every channel's ligand follows a program of its sender type
        chs = [c for c in channels if (c["sender"], c["receiver"]) == (pr["sender"], pr["receiver"])]
        for m, c in enumerate(chs):
            a, gl = c["sender"], c["lig"]
            kind = "true" if m == 0 else ("inert" if m == cfg.lr_decoys else "coprog")
            c["lig_kind"] = kind
            pi = prog[a]["inert" if kind == "inert" else "active"]["pi"]
            w = 1.0 if kind == "true" else prng.uniform(0.6, 1.4)
            on = ct == a
            rate = np.where(on, cfg.prog_load * w * pi + cfg.lig_own_sd * prng.normal(size=N), 0.0)
            lr_level[gl] = np.where(on, cfg.prog_level + rate, 0.0)
            if kind == "true":
                lig_dose[gl] = np.where(on, np.exp(rate), 0.0)
    # ---- the dose: ligand-weighted kernel over the senders, saturating, times a saturating receptor gate
    h = {}
    for pr in cfg.pairs:
        a, b = pr["sender"], pr["receiver"]
        rb, Ea = ct == b, E[:, a]
        h[(a, b)] = float(np.median(Ea[rb & (Ea > 0)])) if (rb & (Ea > 0)).any() else 1.0
        ch = [c for c in channels if c["true"] and (c["sender"], c["receiver"]) == (a, b)][0]
        L, R = lig_dose[ch["lig"]], lr_level[ch["rec"]]
        on = (ct[src] == a) & (ct[dst] == b)
        EL = np.bincount(dst[on], k[on] * L[src[on]], minlength=N)
        hL = float(np.median(EL[rb & (EL > 0)])) if (rb & (EL > 0)).any() else 1.0
        hR = float(np.median(R[R > 0])) if (R > 0).any() else 1.0
        pair_dose[(a, b)] = np.where(rb, gain(pr, t) * (EL / (EL + hL) * R / (R + hR)), 0.0)
    # ---- decoder and counts
    W = rng.normal(0, 0.15, size=(G, D))
    for kk in range(D):
        W[kk * cfg.module_size:(kk + 1) * cfg.module_size, kk] += cfg.loading
    bias = rng.normal(0, 0.5, size=G)
    markers, lig_gene, rec_gene = np.zeros((N, G)), {}, {}
    for a in range(T):             # every type's marker ligand / receptor
        lig_gene[a], rec_gene[a] = D * cfg.module_size + a, D * cfg.module_size + T + a
        on = (ct == a) & (rng.uniform(size=N) < 0.6)
        markers[on, lig_gene[a]] += rng.lognormal(0, 0.5, on.sum())
        on = (ct == a) & (rng.uniform(size=N) < 0.7)
        markers[on, rec_gene[a]] += rng.lognormal(0, 0.5, on.sum())
    for gene, level in lr_level.items():
        markers[:, gene] += level
    env_genes = list(range(G - cfg.env_genes, G))
    w_env = cfg.env_effect * erng.uniform(0.5, 1.5, cfg.env_genes)
    markers[:, env_genes] += env[:, None] * w_env[None]
    resp_total, M = np.zeros((N, G)), np.zeros((N, T))
    for tt, gs in ident.items():
        markers[np.ix_(ct == tt, gs)] += cfg.type_gene_level
    for a, pa in prog.items():
        on = ct == a
        for kind in ("active", "inert"):
            markers[np.ix_(on, pa[kind]["genes"])] += cfg.prog_level + cfg.prog_load * pa[kind]["pi"][on, None] * pa[kind]["w"][None]
    amp = 0.0 if cfg.null else cfg.resp_effect
    for (a, b), r_ in resp.items():
        on = ct == b
        markers[np.ix_(on, r_["genes"])] += cfg.resp_level + cfg.resp_noise * r_["xi"][on, None] * r_["v"][None]
        ne = int(round(cfg.env_resp_frac * len(r_["genes"])))
        r_["env_genes"] = list(r_["genes"][:ne]) if cfg.env_resp_frac > 0 else []
        if r_["env_genes"]:        # the niche field loads on part of the same pathway
            markers[np.ix_(on, r_["env_genes"])] += cfg.env_resp_effect * env[on, None] * np.sign(r_["v"][:ne])[None]
        delta = amp * pair_dose[(a, b)][:, None] * r_["v"][None]
        resp_total[:, r_["genes"]] += delta
        r_["delta"] = delta.astype(np.float32)
        M[:, a] = np.sqrt(M[:, a] ** 2 + (delta ** 2).sum(1) / G)     # gene-space magnitude of the response
    M = np.sqrt(0.0 + M ** 2)
    logp = _logsoftmax((kappa + niche) @ W.T + bias + markers + resp_total)
    mseed = seed if iv.get("counts") is None else iv["counts"]
    crng = rng if iv.get("counts") is None else np.random.default_rng(iv["counts"])
    lib = cfg.counts_per_cell * np.exp(crng.normal(0, cfg.lib_sd, size=(N, 1)))
    X = crng.negative_binomial(cfg.dispersion, cfg.dispersion / (cfg.dispersion + lib * np.exp(logp))).astype(np.float32)
    # ---- segmentation bleed: a share of every cell's transcripts counted in its nearest cells
    X_clean, brng = X.copy(), np.random.default_rng(mseed + 5_555)
    dist, nn = cKDTree(xy).query(xy, k=cfg.bleed_k + 1)
    w = 1.0 / (1.0 + dist[:, 1:])
    w, nn = w / w.sum(1, keepdims=True), nn[:, 1:]
    Xi = X.astype(np.int64)
    out = brng.binomial(Xi, cfg.bleed)
    bleed_in = np.zeros_like(Xi)
    for j in range(N):
        if out[j].any():
            np.add.at(bleed_in, nn[j], brng.multinomial(out[j], w[j]).T)
    X = (Xi - out + bleed_in).astype(np.float32)
    a_ref = ad.AnnData(X)
    a_ref.obs_names = [str(i) for i in range(N)]
    a_ref.var_names = [f"g{kk}" for kk in range(G)]
    a_ref.obsm["spatial"] = a_ref.obsm["desi_map"] = xy
    a_ref.obs["cell_type"] = ct.astype(str)
    a_ref.obs["cell_type"] = a_ref.obs["cell_type"].astype("category")
    a_ref.obs["t"] = t
    a_q = ad.AnnData(U)
    a_q.obsm["spatial"] = a_q.obsm["xenium_map"] = qxy
    truth = dict(ctype=ct, xy=xy, t=t, z_niche=z_niche, kappa=kappa, mu=mu, niche=niche, M=M, E=E, h=h, pairs=cfg.pairs,
                 W=W, bias=bias, lig_gene=lig_gene, rec_gene=rec_gene, aux_xy=qxy, aux_U=U, lam=cfg.lam, z_pix=z_pix,
                 channels=channels, env=env, env_genes=env_genes, env_w=w_env, X_clean=X_clean, programs=prog, resp=resp,
                 ident=ident, pair_dose=pair_dose, lig_dose=lig_dose, intervene=dict(iv))
    return a_ref, a_q, truth


def make_tissue(scenario="mixed", seed=250, null=False, aux="default", intervene=None):
    """A tissue (and the planted twin's truth): counts, positions, types, the per-edge truth (ligand-weighted share of each
    receiver's response), the LR channel database. aux='noise': the auxiliary modality replaced by seeded noise."""
    cfg = Config(**copy.deepcopy(SCENARIOS[scenario]))
    ref, aux_ad, trp = simulate(cfg, seed, intervene)
    tr = trp
    if null:
        cfg.null = True
        ref, aux_ad, tr = simulate(cfg, seed, intervene)
    if aux == "noise":
        aux_ad = ad.AnnData(np.random.default_rng(424_242 + seed).standard_normal(aux_ad.shape).astype(np.float32),
                            obsm=dict(spatial=aux_ad.obsm["spatial"], xenium_map=aux_ad.obsm["xenium_map"]))
    t = SimpleNamespace(scenario=scenario, seed=seed, null=null, aux=aux, config=cfg, adata_ref=ref, adata_aux=aux_ad, truth=tr,
                        planted=trp)
    t.ct, t.xy, t.T, t.G, t.radius, t.M = trp["ctype"].astype(int), trp["xy"], cfg.n_types, cfg.n_genes, cfg.graph_radius, trp["M"]
    pr = np.array(sorted(cKDTree(t.xy).query_pairs(r=cfg.graph_radius, output_type="set")))
    src, dst = np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]]
    keep = t.ct[src] != t.ct[dst]
    o = np.lexsort((src[keep], dst[keep]))
    t.src, t.dst = src[keep][o], dst[keep][o]
    k = np.exp(-np.linalg.norm(t.xy[t.src] - t.xy[t.dst], axis=1) / lam_edges(cfg, t.ct[t.src], t.ct[t.dst]))
    w = np.zeros(len(t.src))
    lig_of = {(c["sender"], c["receiver"]): c["lig"] for c in trp["channels"] if c["true"]}
    for (a_, b_), gl in lig_of.items():
        on = (t.ct[t.src] == a_) & (t.ct[t.dst] == b_)
        w[on] = k[on] * trp["lig_dose"][gl][t.src[on]]
    key = t.dst * t.T + t.ct[t.src]
    on_lr = np.isin(t.ct[t.src] * t.T + t.ct[t.dst], [a_ * t.T + b_ for (a_, b_) in lig_of])
    t.strength = np.where(on_lr, t.M[t.dst, t.ct[t.src]] * w / np.maximum(np.bincount(key, w, minlength=t.T * len(t.ct))[key], EPS),
                          t.M[t.dst, t.ct[t.src]] * k / np.maximum(trp["E"][t.dst, t.ct[t.src]], EPS))
    t.channels = trp["channels"]
    lg, rg = trp["lig_gene"], trp["rec_gene"]
    t.lig_type = {int(g): a for a, g in lg.items()} | {c["lig"]: c["sender"] for c in t.channels}
    t.rec_type = {int(g): b for b, g in rg.items()} | {c["rec"]: c["receiver"] for c in t.channels}
    t.lig_db, t.rec_db = sorted(t.lig_type), sorted(t.rec_type)
    return t


def sender_genes(trp, T):
    """Each type's identity, program and ligand genes (where its transcripts could bleed from)."""
    out = {tt: set(int(g) for g in trp["ident"].get(tt, [])) for tt in range(T)}
    for tt, pa in trp["programs"].items():
        out[tt] |= set(pa["active"]["genes"]) | set(pa["inert"]["genes"])
    out = {tt: sorted(v) for tt, v in out.items()}
    for c in trp["channels"]:
        out[c["sender"]].append(int(c["lig"]))
    return out

# ------------------------------------------------------------------------------------------------ truth read-outs for the benchmark
def niche_fields(t):
    """The planted tissue's niche as the benchmark hands it to the niche tests: its three niche fields and the environment field
    (logit), standardised -> (N, 4)."""
    tr = t.truth
    env = np.asarray(tr["env"], float)
    F = np.column_stack([np.asarray(tr["z_niche"], float), 0.5 * np.log(np.clip(env, 1e-9, 1 - 1e-9) / np.clip(1 - env, 1e-9, 1))])
    return (F - F.mean(0)) / F.std(0)


def planted_effect(t):
    """(T, G) SD over each type's cells of the planted niche-driven log-rate (the niche fields' shift + the environment field's
    loads); nan for types under 30 cells."""
    tr, ct = t.truth, t.ct
    A = lambda x: np.asarray(x, float)
    C = A(tr["niche"]) @ A(tr["W"]).T
    C[:, tr["env_genes"]] += A(tr["env"])[:, None] * A(tr["env_w"])[None]
    for (a, b), x in tr["resp"].items():
        if x["env_genes"]:
            on = ct == b
            C[np.ix_(on, x["env_genes"])] += t.config.env_resp_effect * A(tr["env"])[on, None] * np.sign(A(x["v"])[:len(x["env_genes"])])[None]
    return np.stack([C[ct == b].std(0) if (ct == b).sum() >= 30 else np.full(t.G, np.nan) for b in range(t.T)])


def programs_for(t, n_decoy=20, size=8):
    """A user-given program set for the planted tissue: resp_<a>_<b> (each planted pair's responders), niche_<b> (each type's `size`
    strongest planted niche genes, where a niche is planted) and `n_decoy` decoys of `size` random non-responder genes (seeded)."""
    P = {f"resp_{p['sender']}_{p['receiver']}": [f"g{g}" for g in t.truth["resp"][(p["sender"], p["receiver"])]["genes"]] for p in t.config.pairs}
    used = {g for v in P.values() for g in v}
    if t.config.niche_effect > 0:
        eff = planted_effect(t)
        for b in range(t.T):
            if np.isfinite(eff[b]).any():
                P[f"niche_{b}"] = [f"g{g}" for g in np.argsort(-np.nan_to_num(eff[b]))[:size]]
                used |= set(P[f"niche_{b}"])
    rng = np.random.default_rng(t.seed)
    pool = np.array([f"g{g}" for g in range(t.G) if f"g{g}" not in used])
    for j in range(n_decoy):
        P[f"decoy_{j}"] = list(rng.choice(pool, size, replace=False))
    return P



def export(t, outdir):
    """Files for the baselines: <name>.h5ad (counts, 'typeN' labels, positions), .truth.json (pairs, response genes, the
    environment module, sender genes), .cells.npz (per-cell env, dose, xi), <scenario>_<seed>.channels.json (LR database)."""
    os.makedirs(outdir, exist_ok=True)
    name, trp = f"{t.scenario}_{t.seed}_{'null' if t.null else 'planted'}", t.truth
    a = ad.AnnData(X=np.asarray(t.adata_ref.X, np.float32))
    a.obs["cell_type"] = [f"type{c}" for c in t.ct]
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    a.obsm["spatial"] = np.asarray(t.xy, float)
    a.var_names = [f"g{g}" for g in range(t.G)]
    a.write_h5ad(os.path.join(outdir, name + ".h5ad"))
    pairs = [dict(sender=p["sender"], receiver=p["receiver"], gain=float(p.get("beta0", 1.0)),
                  genes=[int(g) for g in trp["resp"][(p["sender"], p["receiver"])]["genes"]],
                  v=[float(x) for x in trp["resp"][(p["sender"], p["receiver"])]["v"]],
                  env_genes=[int(g) for g in trp["resp"][(p["sender"], p["receiver"])]["env_genes"]]) for p in t.config.pairs]
    json.dump(dict(scenario=t.scenario, seed=t.seed, null=t.null, n_types=t.T, pairs=pairs, env_genes=[int(g) for g in trp["env_genes"]],
                   sender_genes={str(k): v for k, v in sender_genes(trp, t.T).items()}, bleed=float(t.config.bleed)),
              open(os.path.join(outdir, name + ".truth.json"), "w"))
    np.savez_compressed(os.path.join(outdir, name + ".cells.npz"), env=np.asarray(trp["env"], float),
                        **{f"dose_{a_}_{b_}": np.asarray(trp["pair_dose"][(a_, b_)], float) for a_, b_ in trp["pair_dose"]},
                        **{f"xi_{a_}_{b_}": np.asarray(trp["resp"][(a_, b_)]["xi"], float) for a_, b_ in trp["resp"]})
    json.dump(dict(seed=t.seed, channels=[{k: (int(v) if k in ("sender", "receiver", "lig", "rec") else v) for k, v in c.items()} for c in t.channels],
                   lig_db=[int(g) for g in t.lig_db], rec_db=[int(g) for g in t.rec_db],
                   lig_type={str(g): int(t.lig_type[g]) for g in t.lig_db}, rec_type={str(g): int(t.rec_type[g]) for g in t.rec_db}),
              open(os.path.join(outdir, f"{t.scenario}_{t.seed}.channels.json"), "w"))
    return name


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="export the scenario's tissues for the baselines (benchmarks/simulation/run_*.py)")
    ap.add_argument("outdir"); ap.add_argument("--scenario", default="mixed", choices=sorted(SCENARIOS)); ap.add_argument("--seeds", default="250-259")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    for s in range(lo, hi + 1):
        for null in (False, True):
            print(export(make_tissue(a.scenario, s, null), a.outdir), flush=True)
