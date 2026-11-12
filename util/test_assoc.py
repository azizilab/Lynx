import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import statsmodels.api as sm
import statsmodels.formula.api as smf

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree
from scipy.stats import chi2, gamma, norm, ttest_rel
from patsy import dmatrix
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_cluster_dynamics


# -----------------------------------------------
#  Statistical tests for cell-cell interaction
# -----------------------------------------------

def test_cci_target(df_int, df_abun, cluster_labels, alpha=0.05, alternative='greater'):
    """
    Paired one-sided t-tests between interaction and abundance scores per cell type
    """
    stats = []
    for sender in cluster_labels:
        t_stat, p_val = ttest_rel(
            df_int[sender], df_abun[sender], alternative=alternative
        )
        stats.append((sender, t_stat, p_val))
    results = pd.DataFrame(stats, columns=["cluster", "t_stat", "p_value"])

    # FDR correction
    try:
        reject, qvals, _, _ = multipletests(results["p_value"], alpha=alpha, method="fdr_bh")
        results["q_value"] = qvals
        results["significant"] = reject
    except ZeroDivisionError:
        print("Warning: Invalid p-values for FDR correction.")
        results["q_value"] = np.nan
        results["significant"] = False

    return results.sort_values("q_value")


def test_cci(
    adata: ad.AnnData, 
    cci_df: pd.DataFrame = None, 
    cluster_labels: List[str] = None, 
    cluster_key: str = None,
    n_bins: int = 50,
    version: str = None,
    tests: dict = None,
    **kwargs,
):
    r"""Post-hoc test for significant CCI -> (interaction strength of the significant pairs, -log10 q), sender x receiver

    version : None, the fitted model's (`set_model_configs(version=...)`, kept in adata.uns['lynx'] by `model.evaluate`;
        'original' for outputs of the published code, which carry none), or
        'revised': the C x C test of `get_test_interactions` (kwargs: its arguments; `tests`: its tables if already
            run); the strength is the pair statistic
        'original': the published test, a paired t-test of the attention summary `cci_df` (default
            `plot.summarize_cell_interaction`) against cell-type abundance along the trajectory
    """
    info = adata.uns.get('lynx', {})
    version = version or info.get('version', 'original')
    cluster_key = cluster_key or info.get('cluster_key', 'cell_type')
    if cluster_labels is None:
        cluster_labels = pd.Categorical(adata.obs[cluster_key]).categories

    if version == 'revised':
        cxc = (tests or get_test_interactions(adata, cluster_key=cluster_key, **kwargs))['cxc']
        cci_df, qval_df = (pd.DataFrame(0., index=cluster_labels, columns=cluster_labels) for _ in range(2))
        for sender, receiver, stat, q in cxc.loc[cxc.called, ['sender', 'receiver', 'stat', 'q']].values:
            cci_df.loc[sender, receiver] = stat  # (row: sender, col: receiver)
            qval_df.loc[sender, receiver] = -np.log10(max(q, np.finfo(float).tiny))
        return cci_df, qval_df

    if cci_df is None:
        from plot import summarize_cell_interaction
        cci_df = summarize_cell_interaction(adata, cluster_key=cluster_key, cluster_labels=cluster_labels)
    n_clusters = len(cluster_labels)
    cci_summary = pd.DataFrame(
        adata.obsm['omega'].copy(),
        index=adata.obs_names,
        columns=cluster_labels
    )

    abundance_summary = pd.DataFrame(
        adata.obsm['abundance'].copy(),
        index=adata.obs_names,
        columns=cluster_labels,  
    )
    
    mask = np.zeros((n_clusters, n_clusters))
    qval = np.zeros_like(cci_df.values)
    for i, target in enumerate(cluster_labels):
        if np.any(target in adata.obs[cluster_key].values):
            cci_dynamics = get_cluster_dynamics(
                adata, cci_summary, cluster_key=cluster_key,
                target_cell_type=target, n_bins=n_bins, show_fig=False,
            )
            abun_dynamics = get_cluster_dynamics(
                adata, abundance_summary, cluster_key=cluster_key,
                target_cell_type=target, n_bins=n_bins, show_fig=False,
            )
            test_res = test_cci_target(
                cci_dynamics, abun_dynamics, 
                cluster_labels=cci_dynamics.columns[1:], 
                alternative='greater'
            )

            for _, row in test_res.iterrows():
                j = pd.Index(cluster_labels).get_loc(row['cluster'])
                if row['significant'] and not np.isnan(row['q_value']):
                    mask[j, i] = 1  # (row: sender, col: receiver)
                    qval[j, i] = -np.log10(row["q_value"]) 

    qval_df = pd.DataFrame(
        qval,
        index=cci_df.index,
        columns=cci_df.columns 
    )

    return cci_df*mask, qval_df


def get_test_interactions(adata, fields=None, programs=None, seed=0, B=500, leak=None, cluster_key=None):
    r"""Interaction tests on a fitted run (`model.evaluate`'s output), one scheme at three levels -> {name: table}

    Per receiver type and gene, a Poisson regression of the counts on the tested term, given the prior niche latent,
    every other type's exposure and contact count, the library size and the segmentation leak (`fit_leak`, returned as
    'leak' and reusable through `leak`); the score of the term, standardised by its null draws, is the gene's z.
      cxc    C x C      does sender type a change receiver type b: the sum of the genes' z^2, p against the statistic's
                        own null draws, BH over pairs
      cxcxg  C x C x G  which genes of b: the gene's z, normal p, BH within the pair, reported where the pair is called
      cxcxp  C x C x P  which programs (`programs`: {name: genes}): the same score on the program's genes jointly
      kxc, kxcxg, kxcxp the same for niche fields (`fields`: an obs column or columns, categorical / integer columns
                        being group labels, or an array); kxcxg_block: per gene over all the fields
    Null of the exposure: the sender map moved rigidly over the tissue (`B` shifts, which keep its autocorrelation).
    Null of a niche field: autocorrelation-matched surrogates of the field (`surrogate_fields`). No gene is filtered.
    """
    ctx = Context(adata, cluster_key, B=B, seed=seed)
    L = fit_leak(ctx.X, ctx.ct, ctx.xy)['L'] if leak is None else leak
    out = dict(zip(('cxc', 'cxcxg', 'cxcxp'), test_exposure(ctx, L, programs, B)))
    F, names = _fields_of(adata, fields)
    if F is not None:
        out.update(zip(('kxc', 'kxcxg', 'kxcxp', 'kxcxg_block'), test_niche(ctx, F, L, programs, B, names=names)))
    out['leak'] = L
    return out


def _fields_of(adata, fields):
    """`fields`: an obs column name or a list of them (categorical / integer columns are group labels), or an array."""
    if fields is None:
        return None, None
    if isinstance(fields, str):
        fields = [fields]
    if isinstance(fields, (list, tuple)) and all(isinstance(f, str) for f in fields):
        cols = [adata.obs[f] for f in fields]
        if len(cols) == 1 and (cols[0].dtype.name == 'category' or cols[0].dtype.kind in 'iub'):
            return cols[0].to_numpy(), None
        return np.column_stack([c.to_numpy(float) for c in cols]), list(fields)
    return np.asarray(fields), None


def consensus(runs, keys, need=4, call='called', stat='stat'):
    """One table over runs (a list of per-run tables with `keys` columns, a statistic and a call): the median statistic, the
    number of runs calling, consensus = called in >= `need` runs, and strength = the median statistic of a consensus entry (0 else)."""
    need = min(need, len(runs))
    c = pd.concat(runs, keys=range(len(runs)), names=['run']).reset_index(level=0)
    agg = {stat: (stat, 'median'), 'runs_called': (call, 'sum')}
    if 'exposed' in c:
        agg['exposed'] = ('exposed', 'first')
    out = c.groupby(keys, sort=False).agg(**agg).reset_index()
    out['runs'], out['consensus'] = len(runs), out.runs_called >= need
    out['strength'] = np.where(out.consensus, out[stat].fillna(0.0), 0.0)
    return out


EPS = 1e-12


def bh(p):
    """Benjamini-Hochberg q-values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]
        o = np.argsort(v)
        r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(v)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


def _knn(xy, k):
    d, nn = cKDTree(xy).query(xy, k=k + 1)
    w = 1.0 / (1.0 + d[:, 1:])
    return nn[:, 1:], w / w.sum(1, keepdims=True)


def clean_profiles(X, ct, xy, k=8, min_iso=20, ratio=0.1):
    """(T, G) expression proportions of each type over its isolated cells (k nearest all its own type), and per type its
    foreign genes (< ratio x the highest other type)."""
    nn = _knn(xy, k)[0]
    P = X / np.maximum(X.sum(1, keepdims=True), 1.0)
    T = int(ct.max()) + 1
    same = (ct[nn] == ct[:, None]).mean(1)
    pure = np.zeros((T, X.shape[1]))
    for b in range(T):
        cb = np.where(ct == b)[0]
        iso = cb[same[cb] >= 1.0]
        if len(iso) < min_iso:
            iso = cb[np.argsort(-same[cb])[:max(min_iso, len(cb) // 4)]]
        pure[b] = P[iso].mean(0)
    foreign = {}
    for b in range(T):
        other = np.delete(pure, b, 0).max(0)
        foreign[b] = np.where((pure[b] < ratio * other) & (other > 1e-3))[0]
    return pure, foreign


def bleed_range(X, ct, xy, ks=(2, 4, 8)):
    """The bleed model's k: the nearest-cell kernel (a cell receiving from its k nearest, or spreading to them) that best
    explains each type's foreign-gene counts."""
    lib = np.maximum(X.sum(1, keepdims=True), 1.0)
    T, N, best = int(ct.max()) + 1, len(X), (-np.inf, ks[0])
    for k in ks:
        nn, w = _knn(xy, k)
        pure, foreign = clean_profiles(X, ct, xy, k)
        rows = np.repeat(np.arange(N), k)
        for W in (csr_matrix((w.ravel(), (rows, nn.ravel())), shape=(N, N)), csr_matrix((w.ravel(), (nn.ravel(), rows)), shape=(N, N))):
            C = np.asarray(W @ X)
            ss_res, ss_tot = 0.0, 0.0
            for b in range(T):
                F, cb = foreign[b], np.where(ct == b)[0]
                if len(F) == 0 or len(cb) < 10:
                    continue
                y = X[np.ix_(cb, F)] - lib[cb] * pure[b, F]
                x = C[np.ix_(cb, F)]
                f = np.clip((x * y).sum() / max((x * x).sum(), 1e-12), 0.0, 0.5)
                ss_res += ((y - f * x) ** 2).sum()
                ss_tot += (y ** 2).sum()
            fit = 1 - ss_res / max(ss_tot, 1e-12)
            if fit > best[0]:
                best = (fit, k)
    return best[1]


# --- Display mask of the C x C x G plots ---

def spillover_mask(pure, foreign, a, b, ratio=3.0):
    """Genes of receiver b next to sender a whose change could be another type's transcripts (True): foreign to b, not
    >= `ratio` x higher in b than in a, or >= `ratio` x higher in a third type than in b. From the clean profiles only."""
    pb = np.maximum(pure[b], 1e-9)
    others = [c for c in range(len(pure)) if c not in (a, b)]
    G = pure.shape[1]
    orat = (pure[others] / pb[None]).max(0) if others else np.zeros(G)
    return np.isin(np.arange(G), foreign[b]) | (pure[b] < ratio * pure[a]) | (orat >= ratio)


def cxcxg_mask(adata, cluster_key=None):
    """Display mask for the C x C x G matrix: rows (sender, receiver), one column per gene, True where `spillover_mask` marks
    the gene for that pair. Never used by a test; plots mark C x C x G entries with it (`plot.disp_interaction_strength`)."""
    ctx = Context(adata, cluster_key, B=0)
    pairs = [(a, b) for b in range(ctx.T) for a in range(ctx.T) if a != b]
    M = np.array([spillover_mask(ctx.pure, ctx.foreign, a, b) for a, b in pairs], bool)
    idx = pd.MultiIndex.from_tuples([(ctx.names[a], ctx.names[b]) for a, b in pairs], names=['sender', 'receiver'])
    return pd.DataFrame(M, index=idx, columns=ctx.genes)


# --- One fitted run on one tissue ---

def _qr(X0):
    return np.linalg.qr(X0)[0]


class Context:
    """One fitted run on one tissue: counts (adata.layers['counts'] if present, else adata.X), positions, types, the prior
    niche latent (adata.obsm['X_z_prior']), the exposures E (saturating), contact counts and ring exposures at any point,
    the 100 um detrending and the B shifts."""

    def __init__(self, adata, cluster_key=None, B=500, seed=0, detrend=100.0, radius=None, alpha=None):
        u = adata.uns['lynx']
        cat = pd.Categorical(adata.obs[cluster_key or u['cluster_key']])
        self.adata, self.names, self.genes = adata, list(cat.categories), list(adata.var_names)
        self.ct = cat.codes.astype(int)
        X = adata.layers['counts'] if 'counts' in adata.layers else adata.X
        X = X.toarray() if hasattr(X, 'toarray') else X
        self.X, self.xy = np.asarray(X, np.float64), np.asarray(adata.obsm['spatial'], np.float64)
        assert (self.X == np.round(self.X)).all(), "The interaction tests need counts in adata.X or adata.layers['counts']"
        self.env = np.asarray(adata.obsm['X_z_prior'])
        self.N, self.G = self.X.shape
        self.T = len(self.names)
        self.radius = float(u['radius'] if radius is None else radius)
        self.alpha = float(u['alpha'] if alpha is None else alpha)
        self.detrend, self.seed, self._cache = detrend, seed, {}
        self.tree = cKDTree(self.xy)
        k = bleed_range(self.X, self.ct, self.xy)
        self.r_con = float(np.median(self.tree.query(self.xy, k=k + 1)[0][:, -1]))
        self.h = None
        self.E, self.E_con, self.E_ring = self.exposures(self.xy, self_excl=True)
        self.pure, self.foreign = clean_profiles(self.X, self.ct, self.xy, k=k)
        self.lo, self.L = self.xy.min(0), self.xy.max(0) - self.xy.min(0)
        rng, sh = np.random.default_rng(seed), []
        while len(sh) < B:
            s = rng.uniform(-self.L / 2, self.L / 2)
            if np.linalg.norm(s) >= 0.25 * float(self.L.min()):
                sh.append(s)
        self.shifts = np.array(sh)

    def wrap(self, P):
        return self.lo + np.mod(P - self.lo, self.L)

    def exposures(self, P, self_excl=False):
        """(E, contacts, ring E) (n, T) at points P: E sums (1 + d)^-alpha over each type's cells within the radius,
        saturating at the observed median; contacts count cells within r_con; the ring sums beyond it."""
        nb = self.tree.query_ball_point(P, r=self.radius)
        ii = np.repeat(np.arange(len(P)), [len(v) for v in nb])
        jj = np.concatenate([np.asarray(v, int) for v in nb]) if len(ii) else np.zeros(0, int)
        d = np.linalg.norm(P[ii] - self.xy[jj], axis=1)
        keep = d > 1e-9 if self_excl else np.ones(len(d), bool)
        ii, jj, d = ii[keep], jj[keep], d[keep]
        w, c = (1.0 + d) ** -self.alpha, self.ct[jj]
        E, Ec, Er = (np.zeros((len(P), self.T)) for _ in range(3))
        np.add.at(E, (ii, c), w)
        con = d <= self.r_con
        np.add.at(Ec, (ii[con], c[con]), 1.0)
        np.add.at(Er, (ii[~con], c[~con]), w[~con])
        if self.h is None:
            self.h = float(np.median(E[E > 0])) if (E > 0).any() else 1.0
            self.h_ring = float(np.median(Er[Er > 0])) if (Er > 0).any() else 1.0
        return E / (E + self.h), Ec, Er / (Er + self.h_ring)

    def smoother(self, cells, scale=None):
        """Row-normalised leave-one-out Gaussian smoother (sd `scale`, default the detrending) over `cells`."""
        scale = self.detrend if scale is None else scale
        if not scale:
            return None
        key = (scale, cells.tobytes())
        if key not in self._cache:
            tr = cKDTree(self.xy[cells])
            D = tr.sparse_distance_matrix(tr, 3 * scale, output_type='coo_matrix')
            off = D.row != D.col
            S = csr_matrix((np.exp(-0.5 * (D.data[off] / scale) ** 2), (D.row[off], D.col[off])), shape=(len(cells), len(cells)))
            self._cache[key] = diags(1.0 / np.maximum(np.asarray(S.sum(1)).ravel(), EPS)) @ S
        return self._cache[key]

    @staticmethod
    def dt(S, Y):
        return Y if S is None else Y - S @ Y

    def exposed(self, a, b, within=50.0):
        """Receivers of type b with a type-a cell within `within` um."""
        ta = self.xy[self.ct == a]
        if not len(ta):
            return 0
        return int(np.isfinite(cKDTree(ta).query(self.xy[self.ct == b], distance_upper_bound=within)[0]).sum())


def _tail(obs, null):
    """Upper-tail p of a sum-of-squares statistic against its own null draws: a gamma fitted to the draws by moments (a sum of squares
    is gamma-like; over many terms it is close to normal, which the gamma also covers); the empirical tail where the fit is undefined."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    emp = (1 + (null >= obs - EPS).sum()) / (1 + len(null))
    if len(null) < 10 or not np.isfinite(obs):
        return float(emp)
    m, v = null.mean(), null.var()
    if m <= 0 or v <= 1e-12 * max(m * m, 1e-12):
        return float(emp)
    return float(gamma.sf(obs, a=m * m / v, scale=v / m))


def _poisson_fit(y, Z, off, iters=25):
    """Poisson IRLS of counts y on Z (with intercept) with log offset off; fitted means."""
    beta = np.zeros(Z.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6)) - np.log(np.exp(off).mean())
    for _ in range(iters):
        eta = Z @ beta + off
        m = np.exp(np.clip(eta, -30, 30))
        z = eta - off + (y - m) / np.maximum(m, 1e-12)
        A = Z.T @ (Z * m[:, None]) + 1e-8 * np.eye(Z.shape[1])
        try:
            nb = np.linalg.solve(A, Z.T @ (m * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(nb - beta)) < 1e-6:
            beta = nb
            break
        beta = nb
    return np.exp(np.clip(Z @ beta + off, -30, 30))


# --- Segmentation leak ---
def _neighbour_ops(xy, ct, R=16):
    """Per rank r = 1..R: 'take in' (cell i <- its r-th nearest cell) and 'spread' (its transpose) as sparse operators, between
    cells of different types only (leak within a type is indistinguishable from shared biology and cannot fake a sender)."""
    nn = cKDTree(xy).query(xy, k=R + 1)[1]
    N, ops = len(xy), []
    for r in range(1, R + 1):
        keep = ct[nn[:, r]] != ct
        A = csr_matrix((np.ones(int(keep.sum())), (np.arange(N)[keep], nn[keep, r])), shape=(N, N))
        ops += [A, A.T.tocsr()]
    return ops


def fit_leak(X, ct, xy, R=16, cycles=500, tol=1e-10):
    """Segmentation leak: each cell's counts = its own expression (type profile x cell size) + a share of its R nearest other-type
    cells' counts, per rank, taken in and spread out; Poisson maximum likelihood by EM with SQUAREM acceleration.
    -> dict(L (N, G) the leak term added to every null mean, k_take, k_spread, s, lam, loglik, iterations, cycles)"""
    X = np.asarray(X.toarray() if hasattr(X, 'toarray') else X, np.float64)
    ct, xy = np.asarray(ct, int), np.asarray(xy, np.float64)
    T = int(ct.max()) + 1
    C = [np.asarray(A @ X) for A in _neighbour_ops(xy, ct, R)]
    Csum = np.array([c.sum() for c in C])
    M, N, G = len(C), X.shape[0], X.shape[1]

    def unpack(th):
        return th[:M], th[M:M + T * G].reshape(T, G), th[M + T * G:]

    def mu_of(th):
        k, lam, s = unpack(th)
        own = s[:, None] * lam[ct]
        leak = np.zeros_like(X)
        for km, Cm in zip(k, C):
            leak += km * Cm
        return own, leak, np.maximum(own + leak, 1e-12)

    def loglik(th):
        mu = mu_of(th)[2]
        return float((X * np.log(mu) - mu).sum())

    def em(th):
        k, lam, s = unpack(th)
        own, _, mu = mu_of(th)
        ratio = X / mu
        k = np.array([km * float((ratio * Cm).sum()) / max(cs, 1e-12) for km, Cm, cs in zip(k, C, Csum)])
        own_n = own * ratio
        lam = np.stack([own_n[ct == c].sum(0) / max(s[ct == c].sum(), 1e-12) for c in range(T)])
        s = own_n.sum(1) / np.maximum(lam[ct].sum(1), 1e-12)
        return np.concatenate([k, lam.ravel(), s])
    s0 = X.sum(1) / X.sum(1).mean()
    lam0 = np.stack([X[ct == c].sum(0) / max(s0[ct == c].sum(), 1e-12) for c in range(T)])
    th = np.concatenate([np.full(M, 0.01), lam0.ravel(), s0])
    cur, ll, evals = loglik(th), [], 0
    for cyc in range(cycles):
        t1 = em(th); t2 = em(t1); evals += 2
        r, v = t1 - th, (t2 - t1) - (t1 - th)
        nv = np.linalg.norm(v)
        a = min(-np.linalg.norm(r) / nv if nv > 0 else -1.0, -1.0)
        tn = em(np.maximum(th - 2 * a * r + a * a * v, 0.0)); evals += 1
        new = loglik(tn)
        if not np.isfinite(new) or new < loglik(t2):
            tn, new = t2, loglik(t2)
        ll.append(new)
        if abs(new - cur) <= tol * abs(new):
            th = tn
            break
        th, cur = tn, new
    k, lam, s = unpack(th)
    return dict(L=mu_of(th)[1], k_take=k[0::2], k_spread=k[1::2], s=s, lam=lam, loglik=np.array(ll), iterations=evals, cycles=cyc + 1)


def _poisson_fit_leak(y, Z, off, add, iters=50):
    """Poisson fit of counts y with mean exp(Z b + off) + add (add >= 0 fixed): Fisher scoring with step halving. -> (own, total mean)"""
    beta = np.zeros(Z.shape[1])
    beta[0] = np.log(max(np.maximum(y - add, 0).mean(), 1e-6)) - np.log(np.exp(off).mean())

    def parts(b):
        own = np.exp(np.clip(Z @ b + off, -30, 30))
        mu = np.maximum(own + add, 1e-12)
        return own, mu, float((y * np.log(mu) - mu).sum())
    own, mu, cur = parts(beta)
    for _ in range(iters):
        g = Z.T @ (own * (y / mu - 1.0))
        A = Z.T @ (Z * (own ** 2 / mu)[:, None]) + 1e-8 * np.eye(Z.shape[1])
        try:
            step = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-4:
            o2, m2, c2 = parts(beta + t * step)
            if c2 >= cur - 1e-12:
                break
            t /= 2
        beta, own, mu, new = beta + t * step, o2, m2, c2
        if np.max(np.abs(t * step)) < 1e-6:
            break
        cur = new
    return own, mu


# --- Surrogate niche fields ---
_SCALES, _MIX = np.geomspace(10.0, 600.0, 16), np.linspace(0.0, 1.0, 21)
_std = lambda v: (v - v.mean(0)) / np.maximum(v.std(0), 1e-12)


def _variogram_pairs(xy, rng, nbins=20, npairs=100000):
    hmax = 0.5 * float(np.ptp(xy, axis=0).max())
    i, j = rng.integers(0, len(xy), npairs), rng.integers(0, len(xy), npairs)
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    ok = (i != j) & (d <= hmax)
    return i[ok], j[ok], np.clip(np.digitize(d[ok], np.linspace(0, hmax, nbins + 1)) - 1, 0, nbins - 1), nbins


def _gamma(F, pairs):
    i, j, idx, nbins = pairs
    return np.bincount(idx, 0.5 * (F[i] - F[j]) ** 2, minlength=nbins) / np.maximum(np.bincount(idx, minlength=nbins), 1)


def _kernel(xy, scale, cap=1200.0):
    tree = cKDTree(xy)
    Dm = tree.sparse_distance_matrix(tree, min(3 * scale, cap), output_type='coo_matrix')
    K = csr_matrix((np.exp(-0.5 * (Dm.data / scale) ** 2), (Dm.row, Dm.col)), shape=(len(xy), len(xy))) + diags(np.ones(len(xy)))
    return diags(1.0 / np.asarray(K.sum(1)).ravel()) @ K


def surrogate_fields(F, xy, B, rng):
    """Autocorrelation-matched surrogates of the columns of F (N, K) over positions xy (Burt et al. 2020, BrainSMASH): Gaussian-
    kernel-smoothed white noise mixed with a nugget, the kernel scale and mixture fitted once per column to its empirical variogram,
    each surrogate rank-matched to the column's values (same marginal distribution). -> (S (B, N, K), recipes [(scale, mix, sse)])"""
    F = np.asarray(F, float)
    F = F[:, None] if F.ndim == 1 else F
    N, K = F.shape
    pairs = _variogram_pairs(xy, rng)
    recipes, kernels = [], {}
    for k in range(K):
        g0, w, best = _gamma(F[:, k], pairs), rng.normal(size=N), (np.inf, None, None)
        for s in _SCALES:
            kernels[s] = kernels.get(s) if s in kernels else _kernel(xy, s)
            smooth = _std(kernels[s] @ w)
            for a in _MIX:
                sse = float(((_gamma(_std(a * smooth + (1 - a) * w), pairs) - g0) ** 2).sum())
                if sse < best[0]:
                    best = (sse, s, a)
        recipes.append((best[1], best[2], best[0]))
    S = np.zeros((B, N, K))
    for b in range(B):
        for k, (s, a, _) in enumerate(recipes):
            w = rng.normal(size=N)
            y = _std(a * _std(kernels[s] @ w) + (1 - a) * w)
            out = np.empty(N)
            out[np.argsort(y)] = np.sort(F[:, k])
            S[b, :, k] = out
    return S, recipes


# --- Score tests ---
def _null_fit(ctx, rb, Xb, Lb, lib, cov, S, R, ok):
    """One null model for a receiver: Q of the design, per gene the working residuals and the information (Poisson, with the leak
    term in the mean when given; the gene's smooth `S @ R` joins the design when S is given)."""
    Zc = np.column_stack([np.ones(len(rb)), cov])
    Q, smooth = _qr(Zc), (S @ R if S is not None else None)
    rw, info = np.zeros_like(Xb), np.zeros_like(Xb)
    for g in np.flatnonzero(ok):
        Zg = Zc if smooth is None else np.column_stack([Zc, smooth[:, g]])
        if Lb is None:
            mu = _poisson_fit(Xb[:, g], Zg, lib)
            rw[:, g], info[:, g] = Xb[:, g] - mu, mu
        else:
            o, mu = _poisson_fit_leak(Xb[:, g], Zg, lib, Lb[:, g])
            rw[:, g], info[:, g] = (Xb[:, g] - mu) * o / mu, o ** 2 / mu
    return dict(Q=Q, rw=rw, info=info)


def _scores(jb, d):
    """Per-gene score of the tested column d (already residualised against the design), standardised by its information; with
    the raw numerator and information for program aggregation. None if d is constant."""
    s = d.std()
    if s < 1e-10:
        return None
    dz = d / s
    num, den = jb['rw'].T @ dz, np.maximum((jb['info'] * (dz ** 2)[:, None]).sum(0), 1e-12)
    return num / np.sqrt(den), num, den


def _receiver(ctx, b, leak, min_cells, min_det, smooth):
    rb = np.where(ctx.ct == b)[0]
    if len(rb) < min_cells:
        return None
    Xb = ctx.X[rb]
    Lb = None if leak is None else np.asarray(leak)[rb]
    Xd = Xb if Lb is None else np.maximum(Xb - Lb, 0.0)
    tot = np.maximum(Xd.sum(1, keepdims=True), 1.0)
    return dict(rb=rb, S=ctx.smoother(rb) if smooth else None, Xb=Xb, Lb=Lb, lib=np.log(tot[:, 0]), R=np.log1p(Xd / tot * 1e4),
                ok=(Xb > 0).mean(0) >= min_det)


def _program_matrix(programs, genes):
    """programs: {name: [gene names]} (membership 1) or a genes x programs DataFrame of weights -> (W (G, P), names); (None, []) if None."""
    if programs is None:
        return None, []
    if isinstance(programs, pd.DataFrame):
        return programs.reindex(genes).fillna(0.0).to_numpy(float), list(programs.columns)
    gi = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(programs)))
    for j, members in enumerate(programs.values()):
        for m in members:
            if m in gi:
                W[gi[m], j] = 1.0
    return W, list(programs)


def _program_scores(num, den, W, ok):
    """The score of one coefficient shared by a program's genes (weights W), over the tested genes."""
    Wo = W * ok[:, None]
    return (Wo.T @ num) / np.sqrt(np.maximum((Wo ** 2).T @ den, 1e-12))


def _standardise(obs, null):
    """z of the observed score against its null draws (rows), with the two-sided normal p; nan where undefined."""
    m, sd = null.mean(0), np.sqrt(np.maximum(null.var(0), 1e-24))
    z = (obs - m) / sd
    return z, (null - m) / sd, 2 * norm.sf(np.abs(z))


def _gene_rows(names, z, p, ok, called, extra):
    q = bh(np.where(ok, p, np.nan))
    return [dict(extra, gene=names[g], z=z[g], p=p[g], q=q[g], significant=bool(q[g] < 0.05) and called) for g in np.flatnonzero(ok)]


def test_exposure(ctx, leak=None, programs=None, B=None, min_cells=30, min_det=0.05, q=0.05):
    """C x C, C x C x G and (with `programs`) C x C x P. Per receiver b and sender a the null model holds the prior niche latent,
    every other type's exposure and contact count, the gene's 100 um smooth and the leak; the tested term is the (saturating)
    exposure to a. z per gene from B shifts of the sender map; pair statistic = sum over tested genes of z^2, p against its own
    shift draws, BH over pairs; genes BH within the pair; programs likewise. -> (cxc, cxcxg, cxcxp) tables; `leak` is
    `fit_leak(...)['L']` (None: no leak term)."""
    B = len(ctx.shifts) if B is None else min(B, len(ctx.shifts))
    T, G = ctx.T, ctx.G
    W, pnames = _program_matrix(programs, ctx.genes)
    Zn_all = _std(ctx.env)
    recv, fits = {}, {}
    for b in range(T):
        r = _receiver(ctx, b, leak, min_cells, min_det, smooth=True)
        if r is None:
            continue
        recv[b] = r
        Kd = ctx.dt(r['S'], Zn_all[r['rb']])
        for a in range(T):
            if a != b:
                others = [c for c in range(T) if c not in (a, b)]
                cov = ctx.dt(r['S'], np.column_stack([Kd, ctx.E[r['rb']][:, others], ctx.E_con[r['rb']][:, others]]))
                fits[(a, b)] = _null_fit(ctx, r['rb'], r['Xb'], r['Lb'], r['lib'], cov, r['S'], r['R'], r['ok'])

    def score(a, b, E):
        r, jb = recv[b], fits[(a, b)]
        d = ctx.dt(r['S'], E[r['rb'], a])
        o = _scores(jb, d - jb['Q'] @ (jb['Q'].T @ d))
        return (None, None) if o is None else (o[0], _program_scores(o[1], o[2], W, r['ok']) if W is not None else None)
    obs = {k: score(k[0], k[1], ctx.E) for k in fits}
    null = {k: np.full((B, G), np.nan) for k in fits}
    nullp = {k: np.full((B, len(pnames)), np.nan) for k in fits}
    for i, s in enumerate(ctx.shifts[:B]):
        E = ctx.exposures(ctx.wrap(ctx.xy + s))[0]
        for k in fits:
            sc, sp = score(k[0], k[1], E)
            if sc is not None:
                null[k][i] = sc
                if sp is not None:
                    nullp[k][i] = sp
    rows_c, genes_by, progs_by = [], {}, {}
    for (a, b), (sc, sp) in obs.items():
        r, ok = recv[b], recv[b]['ok']
        valid = np.isfinite(null[(a, b)][:, 0])
        row = dict(sender=ctx.names[a], receiver=ctx.names[b], stat=np.nan, p=np.nan, exposed=ctx.exposed(a, b), n_genes=int(ok.sum()))
        if sc is not None and valid.sum() >= 10:
            z, zn, p = _standardise(sc, null[(a, b)][valid])
            row.update(stat=float((z[ok] ** 2).sum()), p=_tail(float((z[ok] ** 2).sum()), (zn[:, ok] ** 2).sum(1)))
            genes_by[(a, b)] = (z, p)
            if sp is not None:
                zp_, _, pp = _standardise(sp, nullp[(a, b)][valid])
                progs_by[(a, b)] = (zp_, pp)
        rows_c.append(row)
    cxc = pd.DataFrame(rows_c)
    cxc['q'] = bh(cxc.p.values)
    cxc['called'] = cxc.q < q
    called = {(s, rr): bool(c) for s, rr, c in cxc[['sender', 'receiver', 'called']].values}
    rows_g, rows_p = [], []
    for (a, b), (z, p) in genes_by.items():
        who = dict(sender=ctx.names[a], receiver=ctx.names[b])
        rows_g += _gene_rows(ctx.genes, z, p, recv[b]['ok'], called[(who['sender'], who['receiver'])], who)
        if (a, b) in progs_by:
            zp_, pp = progs_by[(a, b)]
            qp = bh(pp)
            rows_p += [dict(who, program=pnames[j], z=zp_[j], p=pp[j], q=qp[j], n_genes=int(((W[:, j] != 0) & recv[b]['ok']).sum()),
                            significant=bool(qp[j] < q) and called[(who['sender'], who['receiver'])]) for j in range(len(pnames))]
    return cxc, pd.DataFrame(rows_g), pd.DataFrame(rows_p)


def _field_matrix(fields, names=None):
    """A niche: (N,) group labels -> one indicator column per group (names = the labels); (N, K) float columns, standardised."""
    F = np.asarray(fields)
    if F.ndim == 1 and (F.dtype.kind in 'iuObU' or F.dtype == bool):
        cat = pd.Categorical(F)
        return np.column_stack([(cat.codes == k).astype(float) for k in range(len(cat.categories))]), [str(c) for c in cat.categories], True
    F = F[:, None] if F.ndim == 1 else F.astype(float)
    return _std(F), list(names) if names is not None else [f'field_{k}' for k in range(F.shape[1])], False


def test_niche(ctx, fields, leak=None, programs=None, B=None, min_cells=30, min_det=0.05, q=0.05, names=None, seed=None):
    """K x C, K x C x G and (with `programs`) K x C x P for the columns of `fields` ((N, K) continuous niche fields, or (N,) group
    labels, each group one indicator). Per receiver b and field k the null model holds the other fields (continuous fields: each
    tested given the others; groups: one against the rest), every other type's exposure and contact count, the library and the
    leak; no smooth (the niche is itself the large-scale field). z against B autocorrelation-matched surrogates of the field
    (`surrogate_fields`; a parametric null is anti-conservative here: a smooth field aligns by chance with other smooth structure,
    gene by gene); field-x-type statistic = sum of z^2 with p against its own surrogate draws. BH over field x type cells; genes and
    programs BH within the cell, reported where the cell is called. Per (type, gene) also the block over the fields (sum of the gene's
    z^2, p against the same draws, BH within the type, reported where any field of the type is called): the gene-level niche read-out,
    since one smooth field rarely separates from chance gene by gene. -> (kxc, kxcxg, kxcxp, kxcxg_block)."""
    F, fnames, groups = _field_matrix(fields, names)
    T, G, K = ctx.T, ctx.G, F.shape[1]
    W, pnames = _program_matrix(programs, ctx.genes)
    recv, fits = {}, {}
    for b in range(T):
        r = _receiver(ctx, b, leak, min_cells, min_det, smooth=False)
        if r is None:
            continue
        recv[b] = r
        others = [c for c in range(T) if c != b]
        for k in range(K):
            cols = [ctx.E[r['rb']][:, others], ctx.E_con[r['rb']][:, others]] + ([] if groups else [np.delete(F[r['rb']], k, axis=1)])
            fits[(k, b)] = _null_fit(ctx, r['rb'], r['Xb'], r['Lb'], r['lib'], np.column_stack(cols), None, r['R'], r['ok'])

    def score(k, b, Fk):
        r, jb = recv[b], fits[(k, b)]
        d = Fk[r['rb']]
        o = _scores(jb, d - jb['Q'] @ (jb['Q'].T @ d))
        return (None, None) if o is None else (o[0], _program_scores(o[1], o[2], W, r['ok']) if W is not None else None)
    obs = {kb: score(kb[0], kb[1], F[:, kb[0]]) for kb in fits}
    B = len(ctx.shifts) if B is None else B
    S, _ = surrogate_fields(F, ctx.xy, B, np.random.default_rng(ctx.seed if seed is None else seed))
    nulls = {kb: np.full((B, G), np.nan) for kb in fits}
    nullp = {kb: np.full((B, len(pnames)), np.nan) for kb in fits}
    for i in range(B):
        for kb in fits:
            sc, sp = score(kb[0], kb[1], S[i, :, kb[0]])
            if sc is not None:
                nulls[kb][i] = sc
                if sp is not None:
                    nullp[kb][i] = sp
    rows_k, genes_by, zn_by, progs_by = [], {}, {}, {}
    for (k, b), (sc, sp) in obs.items():
        ok = recv[b]['ok']
        row = dict(field=fnames[k], receiver=ctx.names[b], stat=np.nan, p=np.nan, n_genes=int(ok.sum()))
        if sc is not None:
            valid = np.isfinite(nulls[(k, b)][:, 0])
            if valid.sum() >= 10:
                z, zn, p = _standardise(sc, nulls[(k, b)][valid])
                row.update(stat=float((z[ok] ** 2).sum()), p=_tail(float((z[ok] ** 2).sum()), (zn[:, ok] ** 2).sum(1)))
                genes_by[(k, b)], zn_by[(k, b)] = (z, p), zn
                if sp is not None:
                    progs_by[(k, b)] = _standardise(sp, nullp[(k, b)][valid])[::2]
        rows_k.append(row)
    kxc = pd.DataFrame(rows_k)
    kxc['q'] = bh(kxc.p.values)
    kxc['called'] = kxc.q < q
    called = {(f, rr): bool(c) for f, rr, c in kxc[['field', 'receiver', 'called']].values}
    rows_g, rows_p, rows_b = [], [], []
    for (k, b), (z, p) in genes_by.items():
        who = dict(field=fnames[k], receiver=ctx.names[b])
        rows_g += _gene_rows(ctx.genes, z, p, recv[b]['ok'], called[(who['field'], who['receiver'])], who)
        if (k, b) in progs_by:
            zp_, pp = progs_by[(k, b)]
            qp = bh(pp)
            rows_p += [dict(who, program=pnames[j], z=zp_[j], p=pp[j], q=qp[j], n_genes=int(((W[:, j] != 0) & recv[b]['ok']).sum()),
                            significant=bool(qp[j] < q) and called[(who['field'], who['receiver'])]) for j in range(len(pnames))]
    for b in recv:                                                   # the gene's block over the fields: sum of its z^2, p against the same draws
        ks = [k for (k, bb) in genes_by if bb == b]
        if not ks:
            continue
        ok = recv[b]['ok']
        nb = min(zn_by[(k, b)].shape[0] for k in ks)
        blk = np.sum([genes_by[(k, b)][0] ** 2 for k in ks], 0)
        blk_n = np.sum([zn_by[(k, b)][:nb] ** 2 for k in ks], 0)
        pb = np.array([_tail(blk[g], blk_n[:, g]) if ok[g] else np.nan for g in range(G)])
        qb = bh(pb)
        anyf = any(called[(fnames[k], ctx.names[b])] for k in ks)
        rows_b += [dict(receiver=ctx.names[b], gene=ctx.genes[g], stat=blk[g], p=pb[g], q=qb[g], significant=bool(qb[g] < q) and anyf) for g in np.flatnonzero(ok)]
    return kxc, pd.DataFrame(rows_g), pd.DataFrame(rows_p), pd.DataFrame(rows_b)


# --- Ligand-receptor channels ---

def signed_z(z, p, B, trunc=3.0):
    """Calibrated signed z of a gene test: sign(z) Phi^-1(1 - p / 2), p floored at 1 / (B + 1), truncated at +-trunc."""
    p = np.clip(np.asarray(p, float), 1.0 / (B + 1), 1.0)
    return np.clip(np.sign(np.asarray(z, float)) * norm.isf(p / 2), -trunc, trunc)


def ligand_activity(z, testable, potential, rng, K=25, n_null=2000):
    """niche-LR's activity of one ligand: the standardised prior potential's top K genes weight the receiver gene z;
    one-sided p against random K-gene sets of the testable genes. -> (activity, p)"""
    idx = np.where(testable & np.isfinite(z) & np.isfinite(potential))[0]
    if len(idx) < K + 5:
        return np.nan, np.nan
    col = potential[idx]
    sd = col.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, np.nan
    col = (col - col.mean()) / sd
    top = np.argsort(-col, kind='stable')[:K]
    w = col[top] / col[top].mean()
    nw, zz = np.sqrt(np.sum(w ** 2)), z[idx]
    act = float(np.sum(zz[top] * w) / nw)
    null = (zz[np.argsort(rng.random((n_null, len(idx))), axis=1)[:, :K]] * w).sum(1) / nw
    return act, float((1 + np.sum(null >= act)) / (1 + n_null))


def lr_pairs(z, testable, prior, ligands_ok, receptors_ok, db_pairs, gated, rng, n_null=2000, q=0.05):
    """One pair: every candidate ligand's activity and p, BH over the pair's ligands; reported = database pairs (significant
    ligand, receptor in receptors_ok) if the pair is `gated` (called). prior: {ligand index: potential (G,)}."""
    cands = [l for l in prior if l in ligands_ok]
    act, pv = {}, {}
    for l in cands:
        act[l], pv[l] = ligand_activity(z, testable, prior[l], rng, n_null=n_null)
    ok = [l for l in cands if np.isfinite(pv[l])]
    qv = dict(zip(ok, bh([pv[l] for l in ok]))) if ok else {}
    sig = {l for l in ok if qv[l] <= q}
    return dict(activity=act, p=pv, q=qv, significant=sorted(sig),
                reported=[(l, r) for l, r in db_pairs if l in sig and r in receptors_ok] if gated else [])


def test_lr(cxcxg, prior, lr_network, det, cxc=None, B=500, need=4, n_null=2000, seed=0, min_det=0.10, q=0.05, mask=None):
    """Ligand-receptor channels: per pair, each candidate ligand's activity over the receiver genes' calibrated signed z
    (one run, or the median over runs: `cxcxg` a table or a list; genes testable in >= `need` runs), BH over the pair's
    ligands; reported: database pairs (significant ligand, receptor detected in >= min_det of receivers) inside the pairs
    `cxc` calls ('trusted' or 'called'). prior: genes x ligands (ligand-target potentials); det: types x genes detection.
    mask: `cxcxg_mask` ((sender, receiver) x genes): the masked genes' z is 0 (they stay testable, as genes without a
    response), so other types' transcripts cannot make a ligand active."""
    runs = cxcxg if isinstance(cxcxg, (list, tuple)) else [cxcxg]
    genes, types = list(det.columns), list(det.index)
    gi = {g: i for i, g in enumerate(genes)}
    pot = {gi[l]: prior[l].reindex(genes).to_numpy(float) for l in prior.columns if l in gi}
    db = [(gi[l], gi[r]) for l, r in lr_network[['ligand', 'receptor']].values if l in gi and r in gi]
    gate = None
    if cxc is not None:
        col = 'trusted' if 'trusted' in cxc else 'called'
        gate = {(s, r) for s, r, c in cxc[['sender', 'receiver', col]].values if c}
    Z = []
    for t in runs:
        Zt = {}
        for (s, r), x in t.groupby(['sender', 'receiver']):
            v = np.full(len(genes), np.nan)
            v[[gi[g] for g in x.gene]] = signed_z(x.z.values, x.p.values, B)
            Zt[(s, r)] = v
        Z.append(Zt)
    D = det.to_numpy(float)
    rng, rows = np.random.default_rng(seed), []
    for i, s in enumerate(types):
        for j, r in enumerate(types):
            if i == j:
                continue
            zs = np.array([Zt.get((s, r), np.full(len(genes), np.nan)) for Zt in Z])
            with np.errstate(all='ignore'):
                zm = np.nanmedian(zs, 0)
            if mask is not None and (s, r) in mask.index:
                mk = mask.loc[(s, r)].reindex(genes, fill_value=False).to_numpy(bool)
                zm = np.where(mk & np.isfinite(zm), 0.0, zm)
            o = lr_pairs(zm, np.isfinite(zs).sum(0) >= min(need, len(Z)), pot, {l for l in pot if D[i, l] >= min_det},
                         {g for _, g in db if D[j, g] >= min_det}, db, gate is None or (s, r) in gate, rng, n_null, q)
            for l, p_ in o['p'].items():
                if np.isfinite(p_):
                    recs = [genes[rr] for ll, rr in o['reported'] if ll == l]
                    rows.append(dict(sender=s, receiver=r, ligand=genes[l], activity=o['activity'][l], p=p_, q=o['q'][l],
                                     significant=l in o['significant'], receptors=','.join(recs), reported=bool(recs)))
    return pd.DataFrame(rows)


# ----------------------------------------------------
#  statistic tests for trajectory / sex association
# ----------------------------------------------------

def get_feature(df, feature, meta_cols=['sample_id', 'sex', 't', 'zone']):
    r"""Get binned expression of specific feature along w/ metadata"""
    assert feature in df.columns
    feature_df = df[[col for col in meta_cols if col in df.columns]].copy()
    feature_df['expression'] = df[feature].copy()
    return feature_df


def get_bspline(df, dof=5, degree=3):
    assert 't' in df, "Please infer trajectory first"
    
    bs_basis = dmatrix(
        f"bs(t, df={dof}, degree={degree}, include_intercept=False)",
        df, return_type='dataframe'
    )
    spline_cols = ['spline_'+str(i+1) for i in range(bs_basis.shape[1]-1)]
    df[spline_cols] = bs_basis.iloc[:, 1:]  # Drop the intercept term
    return df, spline_cols


def likelihood_ratio_test(full_model, reduced_model, dof):
    LRT_stat = -2 * (reduced_model.llf - full_model.llf)
    pval = chi2.sf(LRT_stat, df=dof)
    return pval


def test_trajectory(df, dof=5, degree=3, likelihood='gaussian'):
    r"""
    Tests if gene expression significantly affected by 
        (1). trajectory (t); (2). sex
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with expression, sample_id, sex, and t columns
    dof : int
        Degrees of freedom for B-spline basis
    degree : int
        Degree of B-spline
    likelihood : str
        'gaussian' for mixed linear models (default)
        'gamma' for Gamma GLM
    """
    assert 't' in df, "Please infer trajectory first"
    assert 'sex' in df and 'sample_id' in df, \
        'Please append sex & sample metadata'
    
    if likelihood not in ['gaussian', 'gamma']:
        raise NotImplementedError(f"Likelihood '{likelihood}' not implemented. Use 'gaussian' or 'gamma'.")
    
    df, spline_terms = get_bspline(df, dof, degree)
    formula1 = " + ".join(spline_terms)
    formula2 = formula1 + " + sex"
    formula3 = f"({formula1}) * sex"
    
    # Fitting multiple models, select w/ the lowest BIC
    if likelihood == 'gaussian':
        null_model = sm.OLS(df['expression'].values, np.ones(len(df))).fit()
        trajectory_model = smf.mixedlm("expression ~ "+formula1, df, groups=df['sample_id']).fit(reml=False)
        sex_model = smf.mixedlm("expression ~ "+formula2, df, groups=df['sample_id']).fit(reml=False)
        interact_model = smf.mixedlm("expression ~ "+formula3, df, groups=df['sample_id']).fit(reml=False)
    
    elif likelihood == 'gamma':
        gamma_family = sm.families.Gamma(link=sm.families.links.Log())
        cov_struct = sm.cov_struct.Exchangeable()
        null_model = smf.gee("expression ~ 1", groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        trajectory_model = smf.gee("expression ~ "+formula1, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        sex_model = smf.gee("expression ~ "+formula2, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()
        interact_model = smf.gee("expression ~ "+formula3, groups=df['sample_id'], data=df, family=gamma_family, cov_struct=cov_struct).fit()


    # Model selection (QICu for gamma/GEE, BIC for gaussian/MixedLM)
    is_trajectory_feature = 0
    is_interact_feature = 0
    sex_coeff = 0.
    sex_pval = 1.

    models = [null_model, trajectory_model, sex_model, interact_model]
    if likelihood == 'gamma':
        common_scale = interact_model.scale
        criteria = [m.qic(scale=common_scale)[1] for m in models]  # QICu
    else:
        criteria = [m.bic for m in models]

    best_idx = int(np.argmin(criteria))
    best_model = models[best_idx]
    if best_idx == 0:
        pass  # stationary dynamics (null / intercept-only)
    else:
        is_trajectory_feature = 1
        if best_idx == 3:
            is_interact_feature = 1
    pred_expr = np.maximum(best_model.predict(), 0)  # Clip to non-negative

    if 'sex[T.M]' in best_model.pvalues:
        sex_pval = sex_model.pvalues['sex[T.M]']
        sex_coeff = sex_model.params['sex[T.M]']

    return pred_expr, (is_trajectory_feature, is_interact_feature, sex_coeff, sex_pval)


def get_test_associations(df, dof=5, degree=3, alpha=.05, likelihood='gaussian'):
    r"""Get test statistics for both trajectory & sex across features
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features, sample_id, sex, and t columns
    dof : int
        Degrees of freedom for B-spline basis
    degree : int  
        Degree of B-spline
    alpha : float
        Significance level for FDR correction
    likelihood : str
        'gaussian' for mixed linear models (default)
        'gamma' for Gamma GLM
    """
    features = df.columns[df.dtypes.apply(lambda x: x.kind in 'if')]  # skip covariate columns
    features = features[:-1]
    fitted_df = df[features].copy()
    fitted_df['sample_id'] = df['sample_id'].copy()
    fitted_df['sex'] = df['sex'].copy()

    pbar = tqdm(range(len(features)))
    cols = ['trajectory_feature', 'interact_feature', 'coeff.sex', 'pval.sex', 'adj-pval.sex']
    test_assoc = np.zeros((len(features), 5))
    test_assoc[:, -1] = np.nan

    for i in pbar:
        feature = df.columns[i]
        feature_df = get_feature(df, feature)
        res = test_trajectory(feature_df, dof, degree, likelihood=likelihood)
        fitted_df[feature] = res[0]
        test_assoc[i, :-1] = res[1]
        pbar.set_description('Feature {0}/{1}'.format(i+1, len(features)))

    # FDR control
    test_assoc_df = pd.DataFrame(test_assoc, index=features, columns=cols)
    pvals = test_assoc_df['pval.sex'].values
    indices = np.where(~np.isnan(pvals))[0]
    test_assoc_df['adj-pval.sex'][indices] = multipletests(pvals[indices], method='fdr_bh')[1]
    
    test_assoc_df['interact_feature'][test_assoc_df['adj-pval.sex'] >= alpha] = 0
    test_assoc_df['interact_feature'] = test_assoc_df['interact_feature'].astype(np.uint8)
    test_assoc_df['trajectory_feature'] = test_assoc_df['trajectory_feature'].astype(np.uint8)
    
    return fitted_df, test_assoc_df

