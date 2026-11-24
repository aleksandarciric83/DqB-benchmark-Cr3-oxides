"""Out-of-fold SHAP attributions (Figures 3 and 10).
For each dataset: 10 x 10 formula-grouped CV (split and model seed = repeat), exact TreeSHAP on the held-out fold.
Importance of a descriptor in a fold = its share (%) of the summed mean |SHAP| over all descriptors; mean and SD over
the 100 folds (Figure 3). For the final dataset, CatBoost and Random Forest SHAP values of every entry are also averaged
over the ten repeats (Figure 10).
Usage: python p1_shap.py <out_dir> <label>=<dataset.xlsx> ... [--fig10 <label>]"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from joblib import Parallel, delayed
from p1lib import data, make, splits

ELECTRON = ["avg_Mulliken EN", "avg_Martynov-Batsanov EN", "avg_First ionization energy (kJ/mol)"]
LOCAL = ["max_metal_ligand_bond_length", "polyhedron volume"]


def shap_fold(model, feats, dtr, ytr, dte, seed):
    import warnings; warnings.filterwarnings("ignore")
    import shap
    m = make(model, feats, seed=seed).fit(dtr, ytr)
    if model == "CatBoost":
        from catboost import Pool
        sv = m.m.get_feature_importance(Pool(dte[feats]), type="ShapValues")[:, :-1]
    else:
        sv = shap.TreeExplainer(m.m).shap_values(dte[feats].values.astype(float))
    return np.asarray(sv)


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]; fig10 = None
    if "--fig10" in args: k = args.index("--fig10"); fig10 = args[k + 1]; args = args[:k] + args[k + 2:]
    rows = []
    for spec in args:
        label, path = spec.split("=", 1); d, feats = data(path); y = d["Dq/B"].values.astype(float); n = len(y)
        models = ["CatBoost"] + (["RandomForest"] if label == fig10 else [])
        tasks = [(m, r, te, tr) for m in models for r in range(10) for tr, te in splits(d, "random", r)]
        res = Parallel(n_jobs=-1, backend="loky")(delayed(shap_fold)(m, feats, d.iloc[tr], y[tr], d.iloc[te], r) for m, r, te, tr in tasks)
        acc = {m: np.zeros((10, n, len(feats))) for m in models}
        shares = []
        for (m, r, te, tr), sv in zip(tasks, res):
            acc[m][r, te, :] = sv
            if m == "CatBoost":
                a = np.abs(sv).mean(0); shares.append(100 * a / a.sum())
        S = np.array(shares)
        for j, f in enumerate(feats): rows.append(dict(dataset=label, descriptor=f, share=S[:, j].mean(), sd=S[:, j].std()))
        grp = {"electron-binding": ELECTRON, "local-structure": LOCAL, "other": [f for f in feats if f not in ELECTRON + LOCAL]}
        for gname, cols in grp.items():
            idx = [feats.index(c) for c in cols]; g = S[:, idx].sum(1)
            rows.append(dict(dataset=label, descriptor="GROUP: " + gname, share=g.mean(), sd=g.std()))
        if label == fig10:
            for m in models:
                np.save(os.path.join(out, f"shap_oof_{label}_{m}.npy"), acc[m].mean(0))
            pd.Series(feats).to_csv(os.path.join(out, "shap_features.csv"), index=False)
        print(label, "done", flush=True)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "shap_shares.csv"), index=False)
    pd.set_option("display.width", 200); print(T.round(2).to_string())
