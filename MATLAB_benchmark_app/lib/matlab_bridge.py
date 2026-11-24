"""Bridge between the MATLAB app and the Python benchmark code of the paper (common185.py).

usage: python matlab_bridge.py <job.json>
job.json: {"code_dir", "data", "out_dir", "scheme", "models", "features", "folds", "repeats", "seed", "states",
           "n_jobs"}
scheme: compositions | spacegroups | sites | kfold | transfer | partition
n_jobs: number of parallel worker processes (-1 = all cores, 1 = serial; default -1). Every
        (model, partition, fold) fit is an independent task with its own seed, so the results do
        not depend on n_jobs. The models themselves stay single-threaded (as in common185.py).
Writes <out_dir>/oof_<model>.csv (repeats x n), std_<model>.csv for GPR, partition_<model>.csv,
and progress.txt (one line per finished model).
Works on Windows, macOS and Linux (worker processes are spawned; hence the __main__ guard)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np

COST = {"svr": 100, "gpr": 30, "catboost": 10, "rf": 8, "xgb": 4, "lgbm": 4}   # submit slow models first


def fit_one(model, feats, dtr, ytr, dte, s):
    """One independent train/predict task (runs in a worker process)."""
    import warnings; warnings.filterwarnings("ignore")
    from common185 import make, SPECS
    m = make(model, feats, seed=s).fit(dtr, ytr)
    p = m.predict(dte)
    sd = m.predict_std(dte) if SPECS[model][0] == "gpr" else None
    return p, sd


def main(jobfile):
    job = json.load(open(jobfile, encoding="utf-8"))
    sys.path.insert(0, job["code_dir"])
    # worker processes must also find common185.py
    os.environ["PYTHONPATH"] = job["code_dir"] + os.pathsep + os.environ.get("PYTHONPATH", "")
    from common185 import load, splits, SPECS
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import r2_score
    from joblib import Parallel, delayed                 # installed with scikit-learn

    data = job["data"]
    if data.lower().endswith(".csv"):                    # common185.load reads xlsx
        import pandas as pd, tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "data.xlsx"); pd.read_csv(data).to_excel(tmp, index=False); data = tmp
    d, feats_all = load(data)
    feats = job.get("features") or feats_all
    y = d["Dq/B"].values.astype(float)
    n, out = len(y), job["out_dir"]
    os.makedirs(out, exist_ok=True)
    K, R, seed = int(job["folds"]), int(job["repeats"]), int(job["seed"])
    n_jobs = int(job.get("n_jobs") or -1)

    def get_splits(scheme, rep):
        st = seed + rep
        if scheme == "compositions":
            g = d.Formula.astype("category").cat.codes.values
            return list(GroupKFold(K, shuffle=True, random_state=st).split(np.zeros(n), groups=g))
        if scheme == "spacegroups":
            return list(GroupKFold(K, shuffle=True, random_state=st).split(np.zeros(n), groups=d["SGR No."].astype(int).values))
        if scheme == "kfold":
            return list(KFold(K, shuffle=True, random_state=st).split(np.zeros(n)))
        if scheme == "sites":
            return splits(d, "cr_site", 0)
        if scheme == "transfer":
            inh = d.Subset.astype(str).str.lower().isin(["kumar", "inherited"]).values
            return [(np.where(inh)[0], np.where(~inh)[0])]
        raise ValueError(scheme)

    models = sorted(job["models"], key=lambda mdl: -COST.get(SPECS[mdl][0], 1))
    tasks, keys = [], []                                 # keys: (model, partition index, test indices)
    if job["scheme"] == "partition":                     # original procedure: 31 ungrouped 10-fold partitions
        states = [int(s) for s in job["states"]]
        for model in models:
            for i, st in enumerate(states):
                for tr, te in KFold(10, shuffle=True, random_state=st).split(np.zeros(n)):
                    keys.append((model, i, te)); tasks.append((model, tr, te, st))
    else:
        reps = 1 if job["scheme"] in ("sites", "transfer") else R
        for model in models:
            for r in range(reps):
                for tr, te in get_splits(job["scheme"], r):
                    keys.append((model, r, te)); tasks.append((model, tr, te, seed + r))

    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_one)(mdl, feats, d.iloc[tr], y[tr], d.iloc[te], s) for mdl, tr, te, s in tasks)

    log = open(os.path.join(out, "progress.txt"), "w")
    if job["scheme"] == "partition":
        S = len(states)
        oof = {mdl: np.zeros((S, n)) for mdl in models}
        r2f = {mdl: [[] for _ in range(S)] for mdl in models}
        for (mdl, i, te), (p, _) in zip(keys, res):
            oof[mdl][i, te] = p; r2f[mdl][i].append(r2_score(y[te], p))
        for mdl in models:
            rows = [[st, np.mean(r2f[mdl][i]), r2_score(y, oof[mdl][i]), np.mean(np.abs(y - oof[mdl][i]))]
                    for i, st in enumerate(states)]
            np.savetxt(os.path.join(out, f"partition_{mdl}.csv"), np.array(rows), delimiter=",")
            log.write(mdl + "\n")
    else:
        oof = {mdl: np.full((reps, n), np.nan) for mdl in models}
        sd = {mdl: np.full((reps, n), np.nan) for mdl in models}
        for (mdl, r, te), (p, s) in zip(keys, res):
            oof[mdl][r, te] = p
            if s is not None: sd[mdl][r, te] = s
        for mdl in models:
            np.savetxt(os.path.join(out, f"oof_{mdl}.csv"), oof[mdl], delimiter=",")
            np.savetxt(os.path.join(out, f"std_{mdl}.csv"), sd[mdl], delimiter=",")
            log.write(mdl + "\n")
    log.write("DONE\n"); log.close()


if __name__ == "__main__":
    main(sys.argv[1])
