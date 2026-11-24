"""Partition selection as in the original procedure (Table 4, Table S9, Figure 5).
Ungrouped ten-fold CV over the 31 random states of the original code; the selected partition has the highest mean
per-fold R2. Model seed = random state (as in the MATLAB bridge).
Usage: python p1_partition.py <out_dir> <label>=<dataset.xlsx>:<models comma-separated> ..."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from p1lib import data, fit_predict, STATES31

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rows = []; per = []
    for spec in sys.argv[2:]:
        label, rest = spec.split("=", 1); path, models = rest.rsplit(":", 1); models = models.split(",")
        d, feats = data(path); y = d["Dq/B"].values.astype(float); n = len(y)
        tasks = [(m, i, tr, te, st) for m in models for i, st in enumerate(STATES31)
                 for tr, te in KFold(10, shuffle=True, random_state=st).split(np.zeros(n))]
        res = Parallel(n_jobs=-1, backend="loky")(delayed(fit_predict)(m, feats, d.iloc[tr], y[tr], d.iloc[te], st) for m, i, tr, te, st in tasks)
        oof = {m: np.zeros((31, n)) for m in models}; pf = {m: [[] for _ in range(31)] for m in models}
        for (m, i, tr, te, st), (p, _) in zip(tasks, res):
            oof[m][i, te] = p; pf[m][i].append(r2_score(y[te], p))
        for m in models:
            pooled = np.array([r2_score(y, oof[m][i]) for i in range(31)]); fold = np.array([np.mean(pf[m][i]) for i in range(31)])
            mae = np.array([np.mean(np.abs(y - oof[m][i])) for i in range(31)]); k = int(np.argmax(fold))
            rows.append(dict(dataset=label, n=n, model=m, state=STATES31[k], sel_pooled=pooled[k], sel_fold=fold[k], sel_MAE=mae[k],
                             avg_pooled=pooled.mean(), avg_pooled_sd=pooled.std(), avg_fold=fold.mean(), avg_MAE=mae.mean(),
                             min_pooled=pooled.min(), max_pooled=pooled.max(), effect_pooled=pooled[k] - pooled.mean(), effect_fold=fold[k] - fold.mean()))
            per += [dict(dataset=label, model=m, state=st, pooled=pooled[i], fold=fold[i], MAE=mae[i]) for i, st in enumerate(STATES31)]
        print(label, "done", flush=True)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "partition_summary.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(out, "partition_per_state.csv"), index=False)
    pd.set_option("display.width", 220); print(T.round(3).to_string())
