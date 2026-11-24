"""Shared helpers for the paper-1 analyses (benchmark of Dq/B prediction on the curated oxide dataset).
Models, splits and metrics come from common185.py (the code of the paper, unchanged)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
from common185 import load, make, splits, metrics, SPECS, MAIN, KUMAR_CB   # noqa: E402

ICF = (2.23, 2.42)
STATES31 = [5, 10, 12, 15, 19, 20, 25, 26, 30, 33, 35, 40, 45, 47, 50, 54, 55, 60, 61, 65, 68, 70, 75, 80, 82, 85, 89, 90, 95, 96, 100]
SCHEMES = {"comp": ("random", 10), "sg": ("space_group", 5), "site": ("cr_site", 1)}


def data(path):
    d, feats = load(path)
    return d.reset_index(drop=True), feats


def regime(v):
    v = np.asarray(v, float)
    return np.where(v < ICF[0], "WCF", np.where(v <= ICF[1], "ICF", "SCF"))


def fit_predict(model, feats, dtr, ytr, dte, seed):
    import warnings; warnings.filterwarnings("ignore")
    m = make(model, feats, seed=seed).fit(dtr, ytr)
    p = m.predict(dte)
    s = m.predict_std(dte) if SPECS[model][0] == "gpr" else None
    return p, s


def oof_scheme(d, feats, models, scheme_key, n_jobs=-1):
    """out-of-fold predictions, shape (reps, n), for every model; seeds as in Table S7 (split and model seed = repeat)."""
    from joblib import Parallel, delayed
    scheme, reps = SCHEMES[scheme_key]
    y = d["Dq/B"].values.astype(float); n = len(y)
    tasks, keys = [], []
    for m in models:
        for r in range(reps):
            for tr, te in splits(d, scheme, r):
                tasks.append((m, tr, te, r)); keys.append((m, r, te))
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_predict)(m, feats, d.iloc[tr], y[tr], d.iloc[te], s) for m, tr, te, s in tasks)
    oof = {m: np.full((reps, n), np.nan) for m in models}; sd = {m: np.full((reps, n), np.nan) for m in models}
    for (m, r, te), (p, s) in zip(keys, res):
        oof[m][r, te] = p
        if s is not None: sd[m][r, te] = s
    return oof, sd


def summarize(y, P):
    rows = [metrics(y, p) for p in P]
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("R2", "MAE", "RMSE")}
    out.update({k + "_sd": float(np.std([r[k] for r in rows])) for k in ("R2", "MAE")})
    return out


def paired_boot(d, y, P, P_ref, nboot=4000, seed=0):
    """paired bootstrap over compositions of the per-entry |error| averaged over repeats (Table S7)."""
    e = np.abs(y - P).mean(0) - np.abs(y - P_ref).mean(0)
    g = d.Formula.astype("category").cat.codes.values; G = np.unique(g)
    idx = [np.where(g == k)[0] for k in G]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(G), len(G)); ii = np.concatenate([idx[k] for k in pick]); vals.append(e[ii].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(e.mean()), float(lo), float(hi)


def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=1, default=float)
