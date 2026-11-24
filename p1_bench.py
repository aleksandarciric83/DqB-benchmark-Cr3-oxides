"""Benchmark (Table S10, Figure 6) and preprocessing sensitivity (Table S8 i, ii).
Usage: python p1_bench.py <dataset.xlsx> <out_dir>"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, numpy as np, pandas as pd
from p1lib import data, oof_scheme, summarize, paired_boot, MAIN

if __name__ == "__main__":
    path, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    d, feats = data(path); y = d["Dq/B"].values.astype(float)
    rows = []; store = {}
    for sk in ("comp", "sg", "site"):
        oof, sd = oof_scheme(d, feats, MAIN + (["CatBoost_catSG", "GPR_raw1/r2"] if sk in ("comp", "sg") else []), sk)
        for m, P in oof.items():
            store[f"{sk}|{m}"] = P
            if m == "GPR": store[f"{sk}|GPR_sd"] = sd[m]
            r = dict(scheme=sk, model=m, **summarize(y, P))
            if m != "Ridge" and m in MAIN:
                r["dMAE_vs_Ridge"], r["lo"], r["hi"] = paired_boot(d, y, P, oof["Ridge"])
                r["sig"] = "▲" if r["hi"] < 0 else ("▼" if r["lo"] > 0 else "")
            rows.append(r)
        print(sk, "done", flush=True)
    T = pd.DataFrame(rows)
    # mean rank by MAE among the seven models (Constant excluded), averaged over schemes
    M7 = [m for m in MAIN if m != "Constant"]
    rk = T[T.model.isin(M7)].copy(); rk["rank"] = rk.groupby("scheme").MAE.rank()
    T = T.merge(rk.groupby("model")["rank"].mean().rename("mean_rank").reset_index(), on="model", how="left")
    T.to_csv(os.path.join(out, "bench_summary.csv"), index=False)
    np.savez(os.path.join(out, "bench_oof.npz"), **{k.replace("/", "_"): v for k, v in store.items()})
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())
