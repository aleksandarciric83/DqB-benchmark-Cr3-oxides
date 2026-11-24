"""Learning curves (Figure 8, Table S8 iv) and nested hyperparameter tuning (Table S8 iii).
Learning curves: formula-grouped ten-fold CV, 5 repeats (split seed = repeat), training part subsampled to a fraction f
with numpy default_rng(1000*repeat + round(100*f)); model seed = repeat.
Nested tuning: unseen-composition CV, repeat 0; inner three-fold grid search (KFold, seed = fold index) on the training
part; compared with the fixed settings of Table 2 on the same splits.
Usage: python p1_lc_nested.py <dataset.xlsx> <out_dir>"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from p1lib import data, make, splits, fit_predict, SPECS, KUMAR_CB

FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LC_MODELS = ["CatBoost", "RandomForest", "SVR", "Ridge", "Constant"]
GRIDS = {
    "CatBoost": dict(depth=[4, 6, 8], l2_leaf_reg=[1.9172, 5], learning_rate=[0.03, 0.09293]),
    "XGBoost": dict(max_depth=[2, 3, 5], learning_rate=[0.03, 0.1], reg_lambda=[1, 5]),
    "LightGBM": dict(num_leaves=[4, 16], min_child_samples=[5, 20], learning_rate=[0.05, 0.1], n_estimators=[200, 500]),
    "RandomForest": dict(max_features=[0.33, 1.0], min_samples_leaf=[1, 3]),
}


def lc_task(m, feats, dtr, ytr, dte, rep, f):
    rng = np.random.default_rng(1000 * rep + int(round(100 * f)))
    k = max(5, int(round(f * len(ytr)))); sub = np.sort(rng.choice(len(ytr), k, replace=False))
    p, _ = fit_predict(m, feats, dtr.iloc[sub], ytr[sub], dte, rep)
    return p, k


def nested_task(name, feats, dtr, ytr, dte, fold):
    import warnings; warnings.filterwarnings("ignore")
    base = make(name, feats, seed=0); kind, kw = SPECS[name]; kw = dict(kw)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        est = CatBoostRegressor(verbose=0, thread_count=1, allow_writing_files=False, random_seed=0, iterations=500)
        X, Xt = dtr[feats], dte[feats]
    elif name == "XGBoost":
        import xgboost as xgb
        est = xgb.XGBRegressor(n_jobs=1, verbosity=0, random_state=0, n_estimators=500, subsample=0.8); X, Xt = dtr[feats].values, dte[feats].values
    elif name == "LightGBM":
        import lightgbm as lgb
        est = lgb.LGBMRegressor(n_jobs=1, verbose=-1, random_state=0, max_depth=3); X, Xt = dtr[feats].values, dte[feats].values
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_jobs=1, random_state=0, n_estimators=500); X, Xt = dtr[feats].values, dte[feats].values
    gs = GridSearchCV(est, GRIDS[name], cv=KFold(3, shuffle=True, random_state=fold), scoring="neg_mean_absolute_error", n_jobs=1).fit(X, ytr)
    return gs.predict(Xt), str(gs.best_params_)


if __name__ == "__main__":
    path, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    d, feats = data(path); y = d["Dq/B"].values.astype(float); n = len(y)
    # ---- learning curves
    tasks = [(m, r, f, tr, te) for m in LC_MODELS for r in range(5) for f in FRACS for tr, te in splits(d, "random", r)]
    res = Parallel(n_jobs=-1, backend="loky")(delayed(lc_task)(m, feats, d.iloc[tr], y[tr], d.iloc[te], r, f) for m, r, f, tr, te in tasks)
    acc = {}
    for (m, r, f, tr, te), (p, k) in zip(tasks, res):
        a = acc.setdefault((m, r, f), [np.full(n, np.nan), []]); a[0][te] = p; a[1].append(k)
    rows = []
    for (m, r, f), (p, ks) in acc.items():
        rows.append(dict(model=m, rep=r, frac=f, n_train=np.mean(ks), R2=r2_score(y, p), MAE=np.mean(np.abs(y - p))))
    LC = pd.DataFrame(rows); LC.to_csv(os.path.join(out, "lc_raw.csv"), index=False)
    S = LC.groupby(["model", "frac"]).agg(n_train=("n_train", "mean"), R2=("R2", "mean"), R2_sd=("R2", "std"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std")).reset_index()
    S.to_csv(os.path.join(out, "lc_summary.csv"), index=False)
    # ---- nested tuning (repeat 0 of unseen compositions)
    sp = splits(d, "random", 0)
    tasks = [(m, j, tr, te) for m in GRIDS for j, (tr, te) in enumerate(sp)]
    res = Parallel(n_jobs=-1, backend="loky")(delayed(nested_task)(m, feats, d.iloc[tr], y[tr], d.iloc[te], j) for m, j, tr, te in tasks)
    fixed = Parallel(n_jobs=-1, backend="loky")(delayed(fit_predict)(m, feats, d.iloc[tr], y[tr], d.iloc[te], 0) for m, j, tr, te in tasks)
    rows = []
    for m in GRIDS:
        pn, pf = np.full(n, np.nan), np.full(n, np.nan); sel = []
        for (mm, j, tr, te), (p, bp), (q, _) in zip(tasks, res, fixed):
            if mm != m: continue
            pn[te] = p; pf[te] = q; sel.append(bp)
        mode = pd.Series(sel).value_counts().index[0]
        rows.append(dict(model=m, R2_fixed=r2_score(y, pf), MAE_fixed=np.mean(np.abs(y - pf)), R2_nested=r2_score(y, pn), MAE_nested=np.mean(np.abs(y - pn)),
                         dMAE=np.mean(np.abs(y - pn)) - np.mean(np.abs(y - pf)), most_frequent=mode))
    NT = pd.DataFrame(rows); NT.to_csv(os.path.join(out, "nested.csv"), index=False)
    pd.set_option("display.width", 220)
    print(S[S.frac.isin([0.2, 0.5, 1.0])].round(3).to_string()); print(NT.round(3).to_string())
