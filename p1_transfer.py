"""Transfer from the inherited entries to the added hosts (Figure 9).
Stage 1: train on all inherited entries, predict all added hosts (stochastic models averaged over seeds 0-4).
Stage 2: k2 added hosts moved to training, 50 partitions, numpy default_rng(s), s = 0..49, model seed s.
Stage 3: t3 added hosts held out (all others in training), 100 partitions, default_rng(10000 + s).choice(added, t3), model seed s
(this is the rule that reproduces the published stage-3 numbers; with 51 added hosts it moves 42 into training).
Usage: python p1_transfer.py <dataset.xlsx> <out_dir> [k2 t3]"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from p1lib import data, fit_predict, MAIN, SPECS

STOCH = {"catboost", "xgb", "lgbm", "rf", "gpr"}

if __name__ == "__main__":
    path, out = sys.argv[1:3]; k2, t3 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (26, 9)
    os.makedirs(out, exist_ok=True)
    d, feats = data(path); y = d["Dq/B"].values.astype(float)
    inh = np.where(d.Subset.astype(str).str.lower().isin(["kumar", "inherited"]))[0]; add = np.where(~d.Subset.astype(str).str.lower().isin(["kumar", "inherited"]))[0]
    parts = {1: [(inh, add, s) for s in range(5)]}
    p2 = []
    for s in range(50):
        rng = np.random.default_rng(s); mv = rng.choice(add, k2, replace=False); p2.append((np.concatenate([inh, mv]), np.setdiff1d(add, mv), s))
    p3 = []
    for s in range(100):
        rng = np.random.default_rng(10000 + s); te = rng.choice(add, t3, replace=False); p3.append((np.concatenate([inh, np.setdiff1d(add, te)]), np.sort(te), s))
    parts[2], parts[3] = p2, p3
    tasks = []
    for st, P in parts.items():
        for m in MAIN:
            for j, (tr, te, s) in enumerate(P):
                if st == 1 and SPECS[m][0] not in STOCH and s > 0: continue
                tasks.append((st, m, j, tr, te, s))
    res = Parallel(n_jobs=-1, backend="loky")(delayed(fit_predict)(m, feats, d.iloc[tr], y[tr], d.iloc[te], s) for st, m, j, tr, te, s in tasks)
    pred = {}
    for (st, m, j, tr, te, s), (p, _) in zip(tasks, res): pred.setdefault((st, m), []).append((j, te, p))
    rows = []; perhost = []
    for (st, m), L in pred.items():
        if st == 1:
            te = L[0][1]; p = np.mean([x[2] for x in L], 0)
            rows.append(dict(stage=1, model=m, n_train=len(inh), n_test=len(te), R2=r2_score(y[te], p), MAE=np.mean(np.abs(y[te] - p)), MAE_sd=0.0, MAE_sem=0.0))
            perhost += [dict(stage=1, model=m, Formula=d.Formula[i], y=y[i], pred=v) for i, v in zip(te, p)]
        else:
            ys = np.concatenate([y[te] for _, te, _ in L]); ps = np.concatenate([p for _, _, p in L])
            maes = np.array([np.mean(np.abs(y[te] - p)) for _, te, p in L])
            rows.append(dict(stage=st, model=m, n_train=len(d) - len(L[0][1]), n_test=len(L[0][1]), R2=r2_score(ys, ps), MAE=maes.mean(),
                             MAE_sd=maes.std(), MAE_sem=maes.std(ddof=1) / np.sqrt(len(maes)), maes=maes.tolist()))
            for _, te, p in L: perhost += [dict(stage=st, model=m, Formula=d.Formula[i], y=y[i], pred=v) for i, v in zip(te, p)]
    T = pd.DataFrame(rows)
    const = {st: np.array(T[(T.stage == st) & (T.model == "Constant")].maes.iloc[0]) for st in (2, 3)}
    T["beats_const"] = [np.nan if (r.stage == 1 or r.model == "Constant") else float(np.mean(np.array(r.maes) < const[r.stage])) for r in T.itertuples()]
    T.loc[T.stage == 1, "beats_const"] = [0.0 if (r.model != "Constant" and r.MAE > T[(T.stage == 1) & (T.model == "Constant")].MAE.iloc[0]) else (1.0 if r.model != "Constant" else np.nan) for r in T[T.stage == 1].itertuples()]
    T.drop(columns=["maes"]).to_csv(os.path.join(out, "transfer_summary.csv"), index=False)
    pd.DataFrame(perhost).to_csv(os.path.join(out, "transfer_perhost.csv"), index=False)
    pd.set_option("display.width", 200); print(T.drop(columns=["maes"]).round(3).to_string())
