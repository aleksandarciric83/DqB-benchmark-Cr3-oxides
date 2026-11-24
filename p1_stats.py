"""Dataset statistics (Tables 1, 3, S5), chemical-class shortcut, regime classification and parity numbers (Figure 7),
GPR calibration, out-of-domain hosts and largest residuals.
Usage: python p1_stats.py <final.xlsx> <curated.xlsx> <bench_dir> <out_dir>"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pymatgen.core import Composition
from p1lib import data, make, regime, ICF

ORDER = ["F", "Cl", "B", "P", "Si", "Ge", "Sb", "Ga", "Al"]
COV7 = ["max_metal_ligand_bond_length", "volume_per_atom", "polyhedron volume", "avg_Mulliken EN", "avg_Metallic valence",
        "std_Mendeleev number", "1/r2"]


def rf(f):
    try: return Composition(f).reduced_formula
    except Exception: return f


def cls(f):
    try: els = {e.symbol for e in Composition(f).elements}
    except Exception: return "others"
    for e in ORDER:
        if e in els: return {"F": "F/Cl", "Cl": "F/Cl"}.get(e, e)
    return "others"


def eff(series):
    p = series.value_counts(normalize=True).values; return float(np.exp(-(p * np.log(p)).sum()))


def class_r2(d):
    c = d.Formula.map(cls); mu = d.groupby(c)["Dq/B"].transform("mean"); y = d["Dq/B"]
    return float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum())


if __name__ == "__main__":
    fpath, cpath, bdir, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    d, feats = data(fpath); y = d["Dq/B"].values.astype(float)
    inh = d.Subset.eq("Kumar").values; add = ~inh
    d["rf"] = d.Formula.map(rf); d["site"] = d.Cr_site.astype(str).replace({"Mg2+/Zr4+": "Mg2+", "Zr4+": "Mg2+"})
    R = {}
    # ---------- Table 3
    def block(m):
        s = d[m]; yy = s["Dq/B"]; reg = pd.Series(regime(yy))
        els = set().union(*[{e.symbol for e in Composition(f).elements} for f in s.Formula])
        top3 = s.site.value_counts().iloc[:3].sum() / len(s)
        return {"entries": len(s), "compositions": s.rf.nunique(), "elements": len(els), "space_groups": s["SGR No."].nunique(),
                "eff_space_groups": eff(s["SGR No."]), "sg_le2": int((s["SGR No."].value_counts() <= 2).sum()),
                "sites": s.site.nunique(), "eff_sites": eff(s.site), "top3_sites": top3,
                "mean": yy.mean(), "sd": yy.std(), "median": yy.median(), "q1": yy.quantile(0.25), "q3": yy.quantile(0.75),
                "min": yy.min(), "max": yy.max(), "WCF": int((reg == "WCF").sum()), "ICF": int((reg == "ICF").sum()), "SCF": int((reg == "SCF").sum())}
    R["table3"] = {"inherited": block(inh), "added": block(add), "final": block(np.ones(len(d), bool))}
    sg_inh = set(d.loc[inh, "SGR No."]); new_sg = d.loc[add & ~d["SGR No."].isin(sg_inh), "SGR No."]
    R["table3"]["new_space_groups"] = int(new_sg.nunique()); R["table3"]["new_sg_entries"] = int(len(new_sg))
    R["table3"]["new_sites"] = sorted(set(d.loc[add, "site"]) - set(d.loc[inh, "site"]))
    # descriptor-space coverage
    cont = [f for f in feats if f != "SGR No."]; X = d[cont].astype(float).copy(); X["1/r2"] = np.log10(X["1/r2"])
    mu, sdv = X[inh].mean(), X[inh].std(); Z = ((X - mu) / sdv).values
    I = np.where(inh)[0]; nn = np.full(len(d), np.nan)
    for i in range(len(d)):
        cand = [j for j in I if j != i and d.rf[j] != d.rf[i]]
        nn[i] = np.min(np.linalg.norm(Z[cand] - Z[i], axis=1))
    thr = np.percentile(nn[inh], 95)
    lo, hi = d.loc[inh, COV7].min(), d.loc[inh, COV7].max()
    outr = ((d[COV7] < lo) | (d[COV7] > hi)).any(axis=1) & add
    R["coverage"] = {"median_nn_inherited": float(np.median(nn[inh])), "median_nn_added": float(np.median(nn[add])), "median_nn_final": float(np.median(nn)),
                     "beyond95_inherited": int((nn[inh] > thr).sum()), "beyond95_added": int((nn[add] > thr).sum()),
                     "outside_range_added": int(outr.sum()), "outside_range_hosts": d.Formula[outr].tolist()}
    # ---------- class shortcut, boundary proximity
    cur, _ = data(cpath)
    R["class_r2"] = {"curated_inherited": class_r2(cur), "added": class_r2(d[add]), "final": class_r2(d)}
    near = (np.minimum(np.abs(y - ICF[0]), np.abs(y - ICF[1])) <= 0.20)
    R["added_near_boundary"] = float(near[add].mean()); R["mean_final"] = float(y.mean())
    R["classes_final"] = d.Formula.map(cls).value_counts().to_dict()
    R["classes_added"] = d[add].Formula.map(cls).value_counts().to_dict()
    # ---------- parity / regimes / calibration from the composition-grouped CV
    Z2 = np.load(os.path.join(bdir, "bench_oof.npz"))
    reg_y = regime(y); maj = pd.Series(reg_y).value_counts(normalize=True).iloc[0]
    R["majority_regime_share"] = float(maj); R["n_ICF"] = int((reg_y == "ICF").sum())
    for m in ["CatBoost", "RandomForest"]:
        P = Z2[f"comp|{m}"]; pm = P.mean(0); fit = make(m, feats, seed=0).fit(d, y).predict(d)
        b = np.polyfit(y, pm, 1)[0]; acc = float((regime(pm) == reg_y).mean()); icf = int(((regime(pm) == "ICF") & (reg_y == "ICF")).sum())
        e = np.abs(y - pm); worst = d.assign(err=e, pred=pm).sort_values("err", ascending=False).head(5)[["Formula", "Dq/B", "pred", "err"]]
        R[m] = {"slope": float(b), "regime_acc": acc, "ICF_hits": icf, "train_R2": float(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "train_MAE": float(np.mean(np.abs(y - fit))), "cv_R2_avgpred": float(1 - ((y - pm) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "cv_MAE_avgpred": float(e.mean()), "worst": worst.round(3).values.tolist(),
                "CdWO4_err": float(e[d.Formula.eq("CdWO4").values][0]), "LiIO3_err": float(e[d.Formula.eq("LiIO3").values][0])}
        pd.DataFrame({"Formula": d.Formula, "y": y, "pred_mean": pm, "pred_sd": P.std(0), "train_fit": fit}).to_csv(os.path.join(out, f"parity_{m}.csv"), index=False)
    P, Sg = Z2["comp|GPR"], Z2["comp|GPR_sd"]; ae = np.abs(y[None, :] - P)
    R["GPR"] = {"cov1": float((ae <= Sg).mean()), "cov2": float((ae <= 2 * Sg).mean()), "spearman": float(spearmanr(Sg.ravel(), ae.ravel()).correlation)}
    for f in ["CdWO4", "LiIO3"]:
        r = d[d.Formula.eq(f)].iloc[0]
        R[f] = {k: float(r[k]) for k in ["Dq/B", "1/r2", "max_metal_ligand_bond_length", "avg_Metallic valence", "std_Mendeleev number"]}
    with open(os.path.join(out, "stats.json"), "w") as fh: json.dump(R, fh, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float)[:6000])
