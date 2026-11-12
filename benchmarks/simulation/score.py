"""Score the simulation benchmark: LYNX by its read-outs (simulation/run_Lynx.py) and every baseline by its own package's test
(run_regression.py, run_AMICI.py, run_NicheDE.py, run_SVCA.py, run_nicheLR.py outputs), on the planted truth, per tissue, mean +- SE over
tissues: C x C, C x C x G, C x C x P (LYNX), ligand-receptor channels through one ligand-target matrix, interaction strength, and in
simulation 1 the niche read-outs (K x C x G / K x C x P for LYNX, the regression F test) against the planted niche effect; in simulation 2
the niche tests' false calls. No mask or filter anywhere.

    python score.py <lynx dir> <baseline dir> [--scenario mixed] [--out SCORE]   -> <out>.json, <out>.md
"""
import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm, f as fdist
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [f"{ROOT}/simulation", f"{ROOT}/util", HERE]
from simulate import make_tissue, planted_effect, Config, SCENARIOS  # noqa: E402
from test_assoc import bh, lr_pairs, signed_z  # noqa: E402
from run_nicheLR import build_db, build_prior  # noqa: E402

FAM = {"LYNX recommended": "lynx_revised", "LYNX recommended, noise auxiliary": "lynx_revised_noiseaux"}
BASE = ["regression", "AMICI (tuned)", "Niche-DE", "Niche-DE γ0.5", "SVCA", "niche-LR (noisy prior)", "niche-LR"]
LR_M = ["LYNX recommended", "LYNX recommended, noise auxiliary", "regression", "AMICI (tuned)", "SVCA"]
STREAM = {"LYNX recommended": 0, "regression": 1, "AMICI (tuned)": 2, "SVCA": 3, "LYNX recommended, noise auxiliary": 4}
SEEDS, T, TAU1 = range(250, 260), 6, 0.25
A = lambda x: np.array(x, float)


def lynx(ro, t):
    """The package's read-outs as a Res: per pair -log10 p and the call; per gene the call, evidence (-log10 p), |z| as the strength
    and the calibrated signed z."""
    r = Res(t.G)
    names, gi = ro["names"], {g: i for i, g in enumerate(ro["genes"])}
    for x in ro["cxc"].itertuples():                       # the pair's evidence for ranking and strength: -log10 p (its own null spread accounted for)
        r.stat[names.index(x.sender), names.index(x.receiver)], r.called[names.index(x.sender), names.index(x.receiver)] = -np.log10(max(x.p, 1e-300)), bool(x.called)
    for (s, rc), x in ro["cxcxg"].groupby(["sender", "receiver"]):
        g = np.array([gi[v] for v in x.gene])
        d = {k: np.zeros(t.G, bool) for k in ("call", "testable")}
        d.update(ev=np.zeros(t.G), r=np.zeros(t.G), z=np.full(t.G, np.nan))
        d["call"][g], d["testable"][g] = x.significant.values, True
        d["ev"][g], d["r"][g] = -np.log10(np.clip(x.p.values, 1e-300, 1)), np.abs(np.nan_to_num(x.z.values))
        d["z"][g] = signed_z(x.z.values, x.p.values, 500)
        r.genes[(names.index(s), names.index(rc))] = d
    return r


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                out[r["tissue"]] = r
    return out


class Res:
    """One method on one tissue: stat / called (T, T); per pair call, ev, testable (G,); edge (tissue edges)."""

    def __init__(self, G):
        self.stat, self.called, self.genes, self.edge, self.G = np.full((T, T), np.nan), np.zeros((T, T), bool), {}, None, G


def regression(rec, G):
    r, q, tv = Res(G), A(rec["q"]), A(rec["t"])          # [receiver b, sender a, gene]
    for a in range(T):
        for b in range(T):
            if a != b:
                c = np.nan_to_num(q[b, a], nan=1.0) < 0.05
                r.stat[a, b], r.called[a, b] = c.sum(), c.any()
                r.genes[(a, b)] = dict(call=c, ev=-np.log10(np.clip(np.nan_to_num(q[b, a], nan=1.0), 1e-300, 1)),
                                       testable=np.isfinite(q[b, a]), z=np.clip(tv[b, a], -3, 3))
    return r


def amici(rec, G):
    r, types, W = Res(G), rec["types"], A(rec["weight"])
    thr, gi = rec.get("tuning", {}).get("weight_threshold_q86", 1.0), {g: i for i, g in enumerate(rec["genes"])}
    for a in range(T):
        for b in range(T):
            if a == b:
                continue
            r.stat[a, b] = W[a, b]
            e = rec["receivers"].get(types[b], {}).get(types[a])
            call, ev, test, z = np.zeros(G, bool), np.zeros(G), np.zeros(G, bool), np.full(G, np.nan)
            for k_, gname in enumerate(rec["receivers"].get(types[b], {}).get("genes", []) if e is not None else []):
                i = gi[gname]
                if e.get("nl10_padj") is not None:
                    pa = e["nl10_padj"][k_]
                    test[i], call[i], ev[i] = np.isfinite(pa), np.isfinite(pa) and pa > -np.log10(0.05), pa if np.isfinite(pa) else 0.0
                if e.get("z") is not None:
                    z[i] = e["z"][k_]
            r.called[a, b] = bool(call.any() and W[a, b] >= thr)      # its guide's interaction graph
            r.genes[(a, b)] = dict(call=call & r.called[a, b], ev=ev, testable=test, z=np.clip(z, -3, 3))
    return r


def nichede(rec, G):
    r, calls, evp, beta = Res(G), np.zeros((T, T, G), bool), np.ones((T, T, G)), A(rec["beta"])   # [index b, niche a, g]
    for lab in ("pos", "neg"):
        gp, cp, ip = (np.nan_to_num(A(rec[lab][k]), nan=1.0) for k in ("gene", "cell_type", "interaction"))
        for b in range(T):
            for a in range(T):
                if a != b:
                    calls[b, a] |= (ip[b, a] < 0.05) & (gp < 0.05) & (cp[:, b] < 0.05)
                    evp[b, a] = np.minimum(evp[b, a], np.maximum.reduce([ip[b, a], gp, cp[:, b]]))
    for a in range(T):
        for b in range(T):
            if a != b:
                r.stat[a, b], r.called[a, b] = calls[b, a].sum(), calls[b, a].any()
                r.genes[(a, b)] = dict(call=calls[b, a], ev=-np.log10(np.clip(evp[b, a], 1e-300, 1)), testable=np.isfinite(beta[b, a]))
    return r


def nichelr(rec, G, prior="noisy"):
    r = Res(G)
    for k, v in rec["res"][prior].items():
        a, b = map(int, k.split("|"))
        act = [x for x in v["activity"].values() if x is not None and np.isfinite(x)]
        r.stat[a, b], r.called[a, b] = (max(act) if act else np.nan), bool(v["lr"])
    return r


def svca(rec, G):
    r, F, P, Q = Res(G), A(rec["frac"]), A(rec["p"]), A(rec["q"])     # [sender, receiver, gene]
    for a in range(T):
        for b in range(T):
            if a != b:
                ok, c = np.isfinite(P[a, b]), np.nan_to_num(Q[a, b], nan=1.0) < 0.05
                r.stat[a, b], r.called[a, b] = (float(np.nanmean(F[a, b])) if ok.any() else np.nan), c.any()
                r.genes[(a, b)] = dict(call=c, ev=np.where(ok, -np.log10(np.clip(np.nan_to_num(P[a, b], nan=1.0), 1e-300, 1)), 0.0),
                                       testable=ok,
                                       z=np.clip(norm.isf(np.clip(P[b, a], 1e-300, 1.0)), -3, 3))     # (as the revision's LR read-out)
    r.meta = rec.get("meta", {})
    return r


def per_tissue(r, t, true_pairs, null):
    """One method on one tissue -> {key: list}: its contribution to every pooled metric."""
    g = defaultdict(list)
    offd = [(i, j) for i in range(T) for j in range(T) if i != j]
    if null:
        g["null pairs"].append(np.mean([r.called[i, j] for i, j in offd]))
        if r.genes:
            g["null genes"].append(sum(int(d["call"].sum()) for d in r.genes.values()))
        return g
    lam = {(p["sender"], p["receiver"]): f"lam{float(p.get('lam', t.config.lam)):g}" for p in t.config.pairs}
    by_range = len(set(lam.values())) > 1                    # several planted ranges: recall per range as well
    for i, j in offd:
        g["score"].append(np.nan_to_num(r.stat[i, j], nan=-1.0)); g["truth"].append((i, j) in true_pairs)
        g["true called" if (i, j) in true_pairs else "false called"].append(r.called[i, j])
        if by_range and (i, j) in true_pairs:
            g["true called " + lam[(i, j)]].append(r.called[i, j])
    if r.genes:
        tp_, fp_, rn = 0, 0, 0
        for (i, j), d in r.genes.items():
            calls = d["call"]
            if (i, j) in true_pairs:
                resp = np.zeros(t.G, bool)
                resp[true_pairs[(i, j)]] = True
                tp_, fp_, rn = tp_ + int((calls & resp).sum()), fp_ + int((calls & ~resp).sum()), rn + int(resp.sum())
                if by_range:
                    g["tp " + lam[(i, j)]].append(int((calls & resp).sum())); g["resp n " + lam[(i, j)]].append(int(resp.sum()))
                tst = d["testable"]
                if resp[tst].any() and (~resp[tst]).any():
                    g["gene auroc"].append(roc_auc_score(resp[tst], d["ev"][tst]))
            else:
                fp_ += int(calls.sum())
        g["tp"].append(tp_); g["fp"].append(fp_); g["resp n"].append(rn)
    if r.edge is not None and np.std(r.edge) > 0:
        on = np.isin(t.ct[t.src] * T + t.ct[t.dst], [a * T + b for a, b in true_pairs])
        g["edge rho"].append(spearmanr(r.edge, t.strength)[0])
        g["edge auroc within"].append(within_receiver_auroc(t.dst, on, r.edge))
        for a, b in true_pairs:
            sel = (t.ct[t.src] == a) & (t.ct[t.dst] == b)
            if sel.sum() >= 20 and np.std(r.edge[sel]) > 0 and np.std(t.strength[sel]) > 0:
                g["edge rho within pair"].append(spearmanr(r.edge[sel], t.strength[sel])[0])
    return g


def sp(x, y):
    x, y = A(x), A(y)
    return float(spearmanr(x, y)[0]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def strength(r, tp_, own):
    """Planted tissue: Spearman of the C x C strength (statistic of a called pair, else 0) with the planted pair strength
    (RMS planted response over the pair's planted responders, `own`), and of the C x C x G strength (|z| for LYNX, the test's
    evidence otherwise, of a significant entry; else 0) with the planted gene shift over those responders."""
    Tg = np.zeros((T, T, tp_.G))
    for (a, b), x in tp_.truth["resp"].items():
        Tg[a, b, np.asarray(x["genes"], int)] = A(x["delta"])[tp_.ct == b].mean(0)
    pairs = sorted(own)
    cxc = sp([np.nan_to_num(r.stat[a, b], nan=0.0) if r.called[a, b] else 0.0 for a, b in pairs],
             [float(np.sqrt(np.mean(Tg[a, b, own[(a, b)]] ** 2))) for a, b in pairs])
    if not r.genes:
        return dict(cxc=cxc, cxg=np.nan)
    xs, ys = [], []
    for a, b in pairs:
        d = r.genes.get((a, b))
        sc = (d["r"] if "r" in d else np.nan_to_num(d["ev"], nan=0.0)) if d is not None else np.zeros(tp_.G)
        call = d["call"] if d is not None else np.zeros(tp_.G, bool)
        xs += list(np.where(call, sc, 0.0)[own[(a, b)]]); ys += list(np.abs(Tg[a, b, own[(a, b)]]))
    return dict(cxc=cxc, cxg=sp(xs, ys))


def lr_tissue(m, r, t, db, dbp, P, det, null):
    """(reported, scores) with the ligand-target matrix from the method's receiver gene z, and its own receptor-gene
    test without the matrix ('(original)')."""
    rng = np.random.default_rng([t.seed, int(null), STREAM[m]])
    rep_w, sc_w, rep_o, sc_o = set(), {}, set(), {}
    lig_db, rec_db = set(db["lig"].astype(int)), set(db["rec"].astype(int))
    for a in range(T):
        for b in range(T):
            if a == b:
                continue
            d = r.genes.get((a, b))
            z = d["z"] if d is not None else np.full(t.G, np.nan)
            test = np.isfinite(z) & (d["testable"] if d is not None and m in FAM else True)
            o = lr_pairs(z, test, P, {g for g in lig_db if det[a, g] >= 0.10}, {g for g in rec_db if det[b, g] >= 0.10}, dbp,
                         bool(r.called[a, b]), rng, n_null=5000)
            rep_w |= {(a, b, l, rr) for l, rr in o["reported"]}
            for l, rr in dbp:
                x = o["activity"].get(l)
                sc_w[(a, b, l, rr)] = x if x is not None and np.isfinite(x) else -1e9
                if d is not None:
                    sc_o[(a, b, l, rr)] = float(np.nan_to_num(d["ev"][rr], nan=-1e9)) if d["testable"][rr] else -1e9
                    if d["call"][rr]:
                        rep_o.add((a, b, l, rr))
                else:
                    sc_o[(a, b, l, rr)] = -1e9
    return (rep_w, sc_w), (rep_o, sc_o)


def lr_nichelr(rec, dbp):
    rep, sc = set(), {}
    for k, x in rec["res"]["oracle"].items():
        a, b = map(int, k.split("|"))
        rep |= {(a, b, int(l[1:]), int(rr[1:])) for l, rr in x["lr"]}
        for l, rr in dbp:
            s_ = x["activity"].get(f"g{l}")
            sc[(a, b, l, rr)] = s_ if s_ is not None and np.isfinite(s_) else -1e9
    return rep, sc


def q1_scores(ev, calls, eff):
    rhos, aucs, sig, c = [], [], [], dict(tp=0, fp=0, fn=0)
    for b in range(T):
        e, f = ev[b], eff[b]
        ok = np.isfinite(e) & np.isfinite(f)
        if not ok.any():
            continue
        if ok.sum() > 10 and np.std(e[ok]) > 0:
            rhos.append(spearmanr(e[ok], f[ok])[0])
        lo, hi = np.percentile(f[ok], [25, 75])
        y = np.where(f >= hi, 1, np.where(f <= lo, 0, -1))
        m_ = ok & (y >= 0)
        if (y[m_] == 1).any() and (y[m_] == 0).any():
            aucs.append(roc_auc_score(y[m_], e[m_]))
        cb, strong = calls[b] & ok, ok & (f >= TAU1)
        c["tp"] += int((cb & strong).sum()); c["fp"] += int((cb & ~strong).sum()); c["fn"] += int((~cb & strong).sum())
        if cb.sum() > 10 and np.std(e[cb]) > 0 and np.std(f[cb]) > 0:
            sig.append(spearmanr(e[cb], f[cb])[0])
    return dict(rho_eff=float(np.median(rhos)) if rhos else np.nan, auroc_q=float(np.median(aucs)) if aucs else np.nan,
                rho_eff_sig=float(np.median(sig)) if sig else np.nan, **c)


def regression_niche(t, eff):
    """The regression baseline with the measured aux channels (mean over the aux pixels within the bigraph radius) as
    covariates: per type and gene the joint F test, BH within the type. -> (F per type x gene, calls)"""
    X, lab, xy = np.asarray(t.adata_ref.X, float), t.ct, t.xy
    Up, tree = np.asarray(t.adata_aux.X, float), cKDTree(np.asarray(t.adata_aux.obsm["spatial"], float))
    near, nn = tree.query_ball_point(xy, r=float(t.config.r_bigraph)), tree.query(xy)[1]
    F_cov = np.stack([Up[p].mean(0) if len(p) else Up[j] for p, j in zip(near, nn)])
    E = np.zeros((len(xy), T))
    for i, js in enumerate(cKDTree(xy).query_ball_point(xy, r=50.0)):
        for j in js:
            if j != i:
                E[i, lab[j]] += 1
    tot = X.sum(1)
    Y, G, k = np.log1p(X / np.maximum(tot, 1)[:, None] * np.median(tot)), X.shape[1], F_cov.shape[1]
    Fv, calls = np.full((T, G), np.nan), np.zeros((T, G), bool)
    for b in range(T):
        sel = lab == b
        if sel.sum() < 30:
            continue
        D0 = np.column_stack([np.ones(sel.sum()), E[sel][:, [c for c in range(T) if c != b]]])
        D1 = np.column_stack([D0, F_cov[sel]])
        r1 = ((Y[sel] - D1 @ np.linalg.lstsq(D1, Y[sel], rcond=None)[0]) ** 2).sum(0)
        r0 = ((Y[sel] - D0 @ np.linalg.lstsq(D0, Y[sel], rcond=None)[0]) ** 2).sum(0)
        dof = sel.sum() - D1.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            Fv[b] = np.where(r1 > 0, ((r0 - r1) / k) / (r1 / dof), np.nan)
        p = fdist.sf(Fv[b], k, dof)
        ok = np.isfinite(p)
        if ok.any():
            calls[b, ok] = bh(p[ok]) < 0.05
    return Fv, calls


def msd(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    return dict(mean=float(v.mean()) if len(v) else float("nan"), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"), n=int(len(v)), values=[float(x) for x in v])


def prf(tp, fp, fn):
    return dict(precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan, f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)


def tissue_metrics(p):
    """per planted tissue from per_tissue's lists"""
    d = {}
    tr, sc = np.array(p["truth"], bool), np.array(p["score"], float)
    d["cxc_auroc"] = float(roc_auc_score(tr, sc)) if tr.any() and (~tr).any() else np.nan
    tc, fc = np.array(p["true called"], bool), np.array(p["false called"], bool)
    d["true_pairs_called"] = float(tc.mean())
    d["cxc_precision"] = float(tc.sum() / (tc.sum() + fc.sum())) if tc.sum() + fc.sum() else np.nan
    d["cxc_f1"] = float(2 * tc.sum() / (2 * tc.sum() + fc.sum() + (~tc).sum())) if tc.sum() + fc.sum() else 0.0
    if p.get("tp"):
        tp, fp, rn = sum(p["tp"]), sum(p["fp"]), sum(p["resp n"])
        d.update(recall=tp / max(rn, 1), precision=tp / (tp + fp) if tp + fp else np.nan, f1=2 * tp / max(2 * tp + fp + (rn - tp), 1) if tp + fp else 0.0,
                 gene_auroc=float(np.median(p["gene auroc"])) if p.get("gene auroc") else np.nan)
    for k in [k for k in p if k.startswith("true called lam")]:          # per planted range: true pairs called, responders recalled
        d["true_pairs_called_" + k.split()[-1]] = float(np.mean(p[k]))
    for k in [k for k in p if k.startswith("resp n lam")]:
        d["recall_" + k.split()[-1]] = sum(p["tp " + k.split()[-1]]) / max(sum(p[k]), 1)
    return d


def programs_tissue(ro, t, eff, gi):
    """LYNX's program level on one tissue: C x C x P (positive = the pair's own planted program), and on a planted tissue with a niche
    K x C x P / K x C x G against the planted effect (programs: RMS SD >= TAU1 over the member genes; genes: SD >= TAU1); null twins:
    calls."""
    out = {}
    X = ro["cxcxp"]
    own = X.program == ("resp_" + X.sender.astype(str) + "_" + X.receiver.astype(str))
    if t.null:
        out["cxcxp_null_calls"] = int(X.significant.sum())
    else:
        tp, fn, fp = int((X.significant & own).sum()), int((~X.significant & own).sum()), int((X.significant & ~own).sum())
        m = prf(tp, fp, fn); out.update(cxcxp_precision=m["precision"], cxcxp_recall=m["recall"], cxcxp_f1=m["f1"])
        z = np.nan_to_num(np.abs(X.z.values)); out["cxcxp_auroc"] = float(roc_auc_score(own.values, z)) if own.any() and (~own).any() else np.nan
    if "kxcxg" not in ro:
        return out
    k = ro["kxc"]                                                   # K x C: the field x type cells and the types called (any field)
    if eff is None:
        out["kxc_false_types"], out["kxc_false_cells"] = k.receiver[k.called].nunique() / k.receiver.nunique(), float(k.called.mean())
    elif not t.null:
        out["kxc_types_called"], out["kxc_cells_called"] = k.receiver[k.called].nunique() / k.receiver.nunique(), float(k.called.mean())
    W = ro["programs"]
    for key, tab, is_prog in (("kxcxp", ro["kxcxp"], True), ("kxcxg", ro["kxcxg_block"], False)):
        unit = "program" if is_prog else "gene"
        if is_prog:                                                 # programs: any field calls it; evidence = max |z| over fields
            agg = tab.assign(za=tab.z.abs()).groupby(["receiver", unit]).agg(call=("significant", "any"), zmax=("za", "max")).reset_index()
        else:                                                       # genes: the block over the fields
            agg = tab.rename(columns={"significant": "call", "stat": "zmax"})[["receiver", unit, "call", "zmax"]]
        if eff is None:                                          # no niche planted: every call is false
            out[f"{key}_false_calls"] = int(agg.call.sum()); continue
        if t.null:
            out[f"{key}_null_calls"] = int(agg.call.sum()); continue
        b = agg.receiver.map(lambda v: ro["names"].index(v)).values
        pe = (np.array([np.sqrt(np.nanmean(eff[bb][[gi[g] for g in W[p]]] ** 2)) for bb, p in zip(b, agg[unit])]) if is_prog
              else np.array([eff[bb, gi[g]] for bb, g in zip(b, agg[unit])]))
        pos, ok = pe >= TAU1, np.isfinite(pe)
        m = prf(int((agg.call & pos).sum()), int((agg.call & ~pos).sum()), int((~agg.call & pos).sum()))
        out.update({f"{key}_{k}": v for k, v in m.items()})
        out[f"{key}_auroc"] = float(roc_auc_score(pos[ok], agg.zmax.values[ok])) if pos[ok].any() and (~pos[ok]).any() else np.nan
        rhos = [spearmanr(agg.zmax.values[(b == bb) & ok], pe[(b == bb) & ok])[0] for bb in np.unique(b) if ((b == bb) & ok).sum() > 10]
        out[f"{key}_rho"] = float(np.median(rhos)) if rhos else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lynx"); ap.add_argument("baselines"); ap.add_argument("--scenario", default="mixed"); ap.add_argument("--out", default=os.path.join(HERE, "SCORE"))
    a = ap.parse_args()
    O = os.path.join(a.baselines, "out")
    REG, AM, SV = load(f"{O}/regression.jsonl"), load(f"{O}/amici_tuned_*.jsonl"), load(f"{O}/svca_*.jsonl")
    ND, ND5, NL = load(f"{O}/nichede_[0-9].jsonl"), load(f"{O}/nichede_g05_*.jsonl"), load(f"{O}/nichelr_*.jsonl")
    niche = Config(**SCENARIOS[a.scenario]).niche_effect > 0          # a niche is planted: score the niche tests against it; else every niche call is false
    per_pl, per_nu, st, nic, prog = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    lr = defaultdict(list)
    for seed in SEEDS:
        tp_ = make_tissue(a.scenario, seed, False)
        true_pairs = {(p["sender"], p["receiver"]): tp_.truth["resp"][(p["sender"], p["receiver"])]["genes"] for p in tp_.config.pairs}
        tdir = os.path.join(a.baselines, "tissues")
        db = build_db(json.load(open(os.path.join(tdir, f"{a.scenario}_{seed}.channels.json"))))
        dbp = list(zip(db["lig"].astype(int), db["rec"].astype(int)))
        ltm = build_prior(db, json.load(open(os.path.join(tdir, f"{a.scenario}_{seed}_planted.truth.json"))), tp_.G, seed, "oracle")
        P = {int(c[1:]): ltm[c].reindex([f"g{g}" for g in range(tp_.G)]).to_numpy(float) for c in ltm.columns}
        true_lr = {(c["sender"], c["receiver"], c["lig"], c["rec"]) for c in tp_.channels if c["true"]}
        eff, gi = (planted_effect(tp_) if niche else None), {f"g{g}": g for g in range(tp_.G)}
        own = {(p["sender"], p["receiver"]): np.array(tp_.truth["resp"][(p["sender"], p["receiver"])]["genes"], int) for p in tp_.config.pairs}
        for null in (False, True):
            t = tp_ if not null else make_tissue(a.scenario, seed, True)
            nm = f"{a.scenario}_{seed}_{'null' if null else 'planted'}"
            det = np.stack([(np.asarray(t.adata_ref.X)[t.ct == c] > 0).mean(0) for c in range(T)])
            ros = {m: pd.read_pickle(f) for m, fam in FAM.items() for f in [os.path.join(a.lynx, fam, nm + ".readouts.pkl")] if os.path.exists(f)}
            res = {m: lynx(ro, t) for m, ro in ros.items()}
            for m, f, rec in (("regression", regression, REG), ("AMICI (tuned)", amici, AM), ("Niche-DE", nichede, ND), ("Niche-DE γ0.5", nichede, ND5), ("SVCA", svca, SV)):
                if nm in rec:
                    res[m] = f(rec[nm], t.G)
            if nm in NL:
                res["niche-LR (noisy prior)"], res["niche-LR"] = nichelr(NL[nm], t.G), nichelr(NL[nm], t.G, "oracle")
            for m, r in res.items():
                (per_nu if null else per_pl)[m].append(per_tissue(r, t, true_pairs, null))
                if not null:
                    st[m].append(strength(r, tp_, own))
            for m, ro in ros.items():
                prog[m].append(programs_tissue(ro, t, eff, gi))
            if not null:
                for m in [x for x in LR_M if x in res]:
                    for row, (rep, sc) in zip((m, f"{m} (original)"), lr_tissue(m, res[m], t, db, dbp, P, det, null)):
                        lab = [k in true_lr for k in sc]
                        lr[row].append(dict(tp=sum(x in rep for x in true_lr), rep=len(rep), pos=len(true_lr), auroc=float(roc_auc_score(lab, list(sc.values()))) if any(lab) and not all(lab) else np.nan))
                if nm in NL:
                    rep, sc = lr_nichelr(NL[nm], dbp); lab = [k in true_lr for k in sc]
                    lr["niche-LR"].append(dict(tp=sum(x in rep for x in true_lr), rep=len(rep), pos=len(true_lr), auroc=float(roc_auc_score(lab, list(sc.values()))) if any(lab) and not all(lab) else np.nan))
            if niche and not null:
                nic["regression"].append(q1_scores(*regression_niche(t, eff), eff))
            elif not niche:
                _, calls = regression_niche(t, np.zeros((T, t.G)))
                nic["regression"].append(dict(false_calls=int(calls.sum())))
        print(a.scenario, seed, flush=True)
    out = {}
    for m in per_pl:
        tm = [tissue_metrics(p) for p in per_pl[m]]
        keys = ["cxc_auroc", "true_pairs_called", "cxc_precision", "cxc_f1", "recall", "precision", "f1", "gene_auroc"] + sorted({k for x in tm for k in x if "_lam" in k})
        d = {k: msd([x.get(k) for x in tm]) for k in keys if any(k in x for x in tm)}
        d["null_pairs_called"] = msd([float(np.mean(p["null pairs"])) for p in per_nu[m]])
        if any(p.get("null genes") for p in per_nu[m]):
            d["null_gene_calls_per_tissue"] = msd([float(p["null genes"][0]) for p in per_nu[m] if p.get("null genes")])
        for k in ("cxc", "cxg"):
            if st.get(m) and any(np.isfinite(e[k]) for e in st[m]):
                d[f"strength_{k}"] = msd([e[k] for e in st[m]])
        out[m] = d
    for m, rows in lr.items():
        d = out.setdefault(m, {})
        tp, rep, pos = sum(r["tp"] for r in rows), sum(r["rep"] for r in rows), sum(r["pos"] for r in rows)
        d["lr_precision"] = msd([r["tp"] / r["rep"] if r["rep"] else np.nan for r in rows]); d["lr_recall"] = msd([r["tp"] / r["pos"] for r in rows])
        d["lr_f1"] = msd([(2 * r["tp"] / (r["rep"] + r["pos"])) if r["rep"] + r["pos"] else 0.0 for r in rows]); d["lr_auroc"] = msd([r["auroc"] for r in rows])
        d["lr_reports"] = dict(value=f"{tp} / {rep}")
    for m, rows in prog.items():
        keys = sorted({k for r in rows for k in r})
        out[m].update({f"prog_{k}": msd([r.get(k, np.nan) for r in rows]) for k in keys})
    for m, rows in nic.items():
        d = out.setdefault(m, {})
        if niche:
            d["niche_rho_eff"] = msd([r["rho_eff"] for r in rows]); d["niche_auroc"] = msd([r["auroc_q"] for r in rows])
            d["niche_precision"] = msd([r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else np.nan for r in rows])
            d["niche_recall"] = msd([r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else np.nan for r in rows])
            d["niche_f1"] = msd([2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"]) if (2 * r["tp"] + r["fp"] + r["fn"]) else 0.0 for r in rows])
        else:
            d["niche_false_calls"] = msd([r["false_calls"] for r in rows])
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=float)
    f2 = lambda e: "–" if not isinstance(e, dict) or "mean" not in e or not np.isfinite(e["mean"]) else f"{e['mean']:.2f} ± {e['se']:.2f}"
    cols = ["cxc_f1", "true_pairs_called", "cxc_precision", "precision", "recall", "f1", "null_gene_calls_per_tissue", "lr_f1", "niche_f1", "niche_false_calls", "prog_cxcxp_f1", "prog_kxcxp_f1", "prog_kxcxg_f1"]
    L = [f"# {a.scenario}: mean ± SE over tissues", "", "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {m} | " + " | ".join(f2(d.get(k)) for k in cols) + " |" for m, d in out.items()]
    open(a.out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
