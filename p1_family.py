"""Family-grouped transfer to the added hosts (Section 3.3, Table S11).
Every added host is predicted once, with all inherited entries in training, under two schemes:
  family : the whole structure family of the host (all added hosts of that family) is held out;
           all added hosts of the other families are in training
  single : only the host itself is held out (all other 182 entries in training)
Families are space groups of the added hosts, with three structure types merged across space groups
(hexaaluminates/hexagallates 186/187/194 without LiIO3, spinels 212/227, double perovskites 14/225 with La2ZnTiO6).
The defined families are fixed before any model is fitted.
Stochastic models: mean over seeds 0-2. Paired dMAE against the constant predictor of the same scheme,
95% CI from 4000 cluster-bootstrap resamples over families.
Usage: python p1_family.py <dataset.xlsx> <out_dir>"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from p1lib import data, fit_predict, MAIN, SPECS

STOCH = {"catboost", "xgb", "lgbm", "rf", "gpr"}
SEEDS = [0, 1, 2]


def family(row):
    sg, f = int(row["SGR No."]), row.Formula
    if f == "LiIO3": return "LiIO3 (194)"
    if sg in (186, 187, 194): return "hexaaluminate/hexagallate"
    if sg in (212, 227): return "spinel"
    if sg == 225 or f == "La2ZnTiO6": return "double perovskite"
    if sg == 230: return "garnet"
    if sg == 55: return "Pbam (Bi2M4O9, Ca3Al2Ge2O10)"
    if sg == 150: return "langasite"
    return f"SG {sg}"


if __name__ == "__main__":
    path, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    d, feats = data(path); y = d["Dq/B"].values.astype(float)
    inh = np.where(d.Subset.astype(str).str.lower().isin(["kumar", "inherited"]))[0]
    add = np.setdiff1d(np.arange(len(d)), inh)
    fam = pd.Series([family(d.iloc[i]) for i in add], index=add)
    splits = {"family": [(np.setdiff1d(np.arange(len(d)), fam.index[fam == F]), fam.index[fam == F].values) for F in fam.unique()],
              "single": [(np.setdiff1d(np.arange(len(d)), [i]), np.array([i])) for i in add]}
    tasks = []
    for sc, S in splits.items():
        for m in MAIN:
            for s in (SEEDS if SPECS[m][0] in STOCH else [0]):
                for tr, te in S: tasks.append((sc, m, s, tr, te))
    res = Parallel(n_jobs=-1, backend="loky")(delayed(fit_predict)(m, feats, d.iloc[tr], y[tr], d.iloc[te], s) for sc, m, s, tr, te in tasks)
    P = {}
    for (sc, m, s, tr, te), (p, _) in zip(tasks, res):
        P.setdefault((sc, m, s), pd.Series(dtype=float))
        P[(sc, m, s)] = pd.concat([P[(sc, m, s)], pd.Series(p, index=te)])
    pred = {}
    for (sc, m, s), ser in P.items(): pred.setdefault((sc, m), []).append(ser.sort_index().loc[add].values)
    pred = {k: np.mean(v, 0) for k, v in pred.items()}
    ya = y[add]; famc = fam.values; F = np.unique(famc); idx = [np.where(famc == k)[0] for k in F]
    rng = np.random.default_rng(0); boot = [np.concatenate([idx[k] for k in rng.integers(0, len(F), len(F))]) for _ in range(4000)]
    rows, ph = [], []
    for sc in splits:
        ec = np.abs(ya - pred[(sc, "Constant")])
        for m in MAIN:
            p = pred[(sc, m)]; e = np.abs(ya - p); de = e - ec
            lo, hi = np.percentile([de[b].mean() for b in boot], [2.5, 97.5]) if m != "Constant" else (np.nan, np.nan)
            rows.append(dict(scheme=sc, model=m, R2=r2_score(ya, p), MAE=e.mean(), dMAE_vs_const=de.mean(), CI_lo=lo, CI_hi=hi,
                             wins_vs_const=float(np.mean(e < ec)) if m != "Constant" else np.nan))
            ph += [dict(scheme=sc, model=m, Formula=d.Formula[i], family=fam[i], y=y[i], pred=v) for i, v in zip(add, p)]
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "family_transfer_summary.csv"), index=False)
    pd.DataFrame(ph).to_csv(os.path.join(out, "family_transfer_perhost.csv"), index=False)
    fam.rename("family").to_frame().assign(Formula=d.Formula[add].values).to_csv(os.path.join(out, "family_definition.csv"))
    pd.set_option("display.width", 200); print(fam.value_counts().to_string()); print(T.round(3).to_string())
