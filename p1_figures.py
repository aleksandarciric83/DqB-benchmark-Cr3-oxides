"""Figures 2-10 of paper 1 from the result files (Figure 1 is produced by original/fig1a.py and does not depend on data).
Usage: python p1_figures.py <res_dir> <final.xlsx> <curated.xlsx> <fig_dir>
Needs in res_dir: bench_summary.csv, bench_oof.npz, partition_summary.csv, partition_per_state.csv, transfer_summary.csv,
transfer_perhost.csv, shap_shares.csv, shap_oof_<final>_*.npy, shap_features.csv, lc_summary.csv, parity_*.csv, stats.json"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from p1lib import data, regime, ICF
from p1_stats import cls

R, FIN, CUR, F = sys.argv[1:5]; os.makedirs(F, exist_ok=True)
d, feats = data(FIN); y = d["Dq/B"].values.astype(float); n = len(d)
cur, _ = data(CUR); n_cur = len(cur); n_add = int((d.Subset == "Added").sum())
S = json.load(open(os.path.join(R, "stats.json")))
NAMES = {"RandomForest": "Random Forest"}
DPI = 600


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".png"), dpi=DPI, bbox_inches="tight"); fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Figure 2 (workflow)
def fig2():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "mathtext.fontset": "stix"})
    T3 = S["table3"]
    cols = [("I", "Data curation", "#1F6FB4", "#EAF2FA",
             f"159 oxide entries\n(Kumar et al.)\n• duplicates merged\n• site mismatches\n  resolved\n• descriptor errors\n  corrected\n\n{n_cur} entries,\n{T3['inherited']['compositions']} compositions"),
            ("II", "Dataset expansion", "#D9541E", "#FCEEE8",
             f"{n_add} Cr$^{{3+}}$-doped\noxide hosts from\nthe literature\n• {T3['new_space_groups']} new space\n  groups\n\n{n} entries,\n{T3['final']['compositions']} compositions,\n{T3['final']['space_groups']} space groups"),
            ("III", "Representation", "#E5A50A", "#FDF6E6", "14 descriptors:\n• composition (6)\n• lattice (4)\n• Cr$^{3+}$ site (3)\n• Cr$^{3+}$ content (1)\n\ntarget: $Dq/B$"),
            ("IV", "Model training", "#7B2D8E", "#F3ECF6", "CatBoost,\nXGBoost,\nLightGBM,\nRandom Forest,\nSVR, GPR\n\nbaselines:\nRidge, constant"),
            ("V", "Cross-validation", "#6AA02B", "#F1F7EA", "grouped by:\n• composition\n  (10×10)\n• space group\n  (5×10)\n• substituted cation\n  (leave-one-out)\n\nR$^2$, MAE, RMSE;\nout-of-fold SHAP"),
            ("VI", "Transfer to\nadded hosts", "#3FB0E6", "#EAF6FC", f"train–test split\nby data source\n\nprogressive split:\n26 → {n_add - 9} of {n_add}\nadded hosts moved\ninto training")]
    fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.set_xlim(0, 10.5); ax.set_ylim(-0.75, 4.4); ax.axis("off")
    w, gap = 1.52, 0.255
    for k, (num, title, c, fill, body) in enumerate(cols):
        x0 = 0.05 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.25), w, 3.35, boxstyle="square,pad=0", fc=fill, ec=c, lw=1.6))
        ax.add_patch(FancyBboxPatch((x0, 3.6), w, 0.75, boxstyle="square,pad=0", fc=c, ec=c, lw=1.6))
        ax.text(x0 + w / 2, 4.17, num, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x0 + w / 2, 3.86 if "\n" not in title else 3.80, title, ha="center", va="center", color="white", fontsize=10.5, linespacing=1.0)
        ax.text(x0 + 0.08, 3.45, body, ha="left", va="top", fontsize=9.4, linespacing=1.5, color="#222222")
        if k < 5:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.03, 2.0), (x0 + w + gap - 0.03, 2.0), arrowstyle="-|>,head_length=4,head_width=2.5", color="#444444", lw=1.1))
    for (xa, xb, lab) in [(0.05, 0.05 + 3 * w + 2 * gap, "Dataset construction (Sections 2.2–2.3)"), (0.05 + 3 * (w + gap), 0.05 + 6 * w + 5 * gap, "Benchmarking and validation (Section 2.4)")]:
        ax.plot([xa, xa, xb, xb], [0.05, -0.05, -0.05, 0.05], color="#888888", lw=0.8)
        ax.text((xa + xb) / 2, -0.38, lab, ha="center", va="center", fontsize=10.5)
    save(fig, "Figure_2_methodological_framework")


# ------------------------------------------------------------------ Figure 4 (distributions)
def fig4():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    kum159 = pd.read_excel(os.path.join(os.path.dirname(FIN), "Kumar_original_oxides_159.xlsx"))
    nonox = pd.read_csv(os.path.join(os.path.dirname(FIN), "nonoxide_33.csv"))
    d192 = pd.concat([kum159[["Formula", "Dq/B"]], nonox[["Formula", "Dq/B"]]], ignore_index=True)
    ORDERC = ["Ga", "B", "P", "Ge", "Sb", "Si", "F/Cl", "Al", "others"]
    LAB = {"Ga": "gallates", "B": "borates", "P": "phosphates", "Ge": "germanates", "Sb": "antimonates", "Si": "silicates", "F/Cl": "fluorides", "Al": "aluminates", "others": "others"}
    BLUE, GREEN, BAND = "#4C78A8", "#2E7D4F", "#D9B8C0"
    bins = np.arange(1.0, 3.75, 0.125)
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 7.0), gridspec_kw={"width_ratios": [1, 1.9], "hspace": 0.62, "wspace": 0.3})
    for row, (D, final) in enumerate([(d192, False), (d, True)]):
        a, b = axs[row]
        a.axvspan(*ICF, color=BAND, alpha=0.8, zorder=0)
        if final:
            a.hist([D[D.Subset == "Kumar"]["Dq/B"], D[D.Subset == "Added"]["Dq/B"]], bins=bins, stacked=True, color=[BLUE, GREEN], edgecolor="white", lw=0.6,
                   label=[f"Kumar et al. ({(D.Subset == 'Kumar').sum()})", f"Added ({(D.Subset == 'Added').sum()})"], zorder=2)
            a.legend(loc="center left", fontsize=7.2, frameon=False, handlelength=1.0, bbox_to_anchor=(0.0, 0.55))
        else:
            a.hist(D["Dq/B"], bins=bins, color=BLUE, edgecolor="white", lw=0.6, zorder=2)
        a.text(0.97, 0.95, f"n = {len(D)}\nmean = {D['Dq/B'].mean():.2f}\nSD = {D['Dq/B'].std():.2f}", transform=a.transAxes, ha="right", va="top", fontsize=8.5)
        for x, t in [(1.85, "WCF"), (2.325, "ICF"), (2.95, "SCF")]: a.text(x, 1.02, t, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
        a.set_xlim(1, 3.6); a.set_xlabel("Dq/B (experimental)"); a.set_ylabel("Count"); a.set_ylim(0, 36)
        for s in ("top", "right"): a.spines[s].set_visible(False)
        a.text(-0.2, 1.08, "ac"[row], transform=a.transAxes, fontsize=12, fontweight="bold")
        b.axhspan(*ICF, color=BAND, alpha=0.8, zorder=0)
        c = D.Formula.map(cls); rng = np.random.default_rng(1)
        for k, cl in enumerate(ORDERC):
            m = (c == cl).values
            if final and cl == "F/Cl":
                b.text(k, 1.07, "excl.", transform=b.get_xaxis_transform(), ha="center", fontsize=7.6, color="#555555"); continue
            if final:
                for sub, mk, col, sz in [("Kumar", "o", BLUE, 14), ("Added", "*", GREEN, 40)]:
                    mm = m & (D.Subset == sub).values
                    b.scatter(k + rng.uniform(-0.22, 0.22, mm.sum()), D["Dq/B"][mm], s=sz, marker=mk, color=col, zorder=3, lw=0)
                na = int((m & (D.Subset == "Added").values).sum())
                b.text(k, 1.03, f"{m.sum()}\n(+{na})", transform=b.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.6, color="#444444")
            else:
                b.scatter(k + rng.uniform(-0.22, 0.22, m.sum()), D["Dq/B"][m], s=14, color=BLUE, zorder=3, lw=0)
                b.text(k, 1.03, f"{m.sum()}", transform=b.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.6, color="#444444")
        b.set_xticks(range(len(ORDERC))); b.set_xticklabels([LAB[c] for c in ORDERC], rotation=40, ha="right", fontsize=8.3); b.set_ylim(1.0, 3.6); b.set_xlim(-0.6, len(ORDERC) - 0.4)
        b.set_ylabel("Dq/B (experimental)")
        b.text(0.0, 1.21 if final else 1.12, "entries per class" + (" (in brackets: added hosts)" if final else ""), transform=b.transAxes, fontsize=7.8, color="#555555")
        bt = b.twinx(); bt.set_ylim(1.0, 3.6); bt.set_yticks([1.85, 2.325, 2.95]); bt.set_yticklabels(["WCF", "ICF", "SCF"], rotation=90, va="center"); bt.tick_params(length=0)
        b.text(-0.12, 1.08, "bd"[row], transform=b.transAxes, fontsize=12, fontweight="bold")
        if final:
            b.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label=f"Kumar et al. ({(D.Subset == 'Kumar').sum()})"),
                              Line2D([], [], marker="*", ls="", ms=9, color=GREEN, label=f"Added ({(D.Subset == 'Added').sum()})")], loc="lower right", fontsize=7.8, frameon=False)
    save(fig, "Figure_4_DqB_distribution")


# ------------------------------------------------------------------ Figure 5 (partition selection)
def fig5():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    P = pd.read_csv(os.path.join(R, "partition_per_state.csv")); T = pd.read_csv(os.path.join(R, "partition_summary.csv"))
    GOLD = "#B8860B"; C = ["#9DB4D3", "#4C78A8", "#1B2F4E"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"wspace": 0.35})
    labs = list(dict.fromkeys(P.dataset)); l0 = labs[0]
    p = P[(P.dataset == l0) & (P.model == "CatBoost")]; t = T[(T.dataset == l0) & (T.model == "CatBoost")].iloc[0]
    a.scatter(p.fold, p.pooled, s=22, facecolor="white", edgecolor="#4C78A8", lw=1.1, label="31 partitions", zorder=3)
    a.scatter([t.sel_fold], [t.sel_pooled], marker="*", s=180, color=GOLD, label="Selected partition", zorder=4)
    a.axhline(t.avg_pooled, color="#4C78A8", ls="--", lw=1); a.axvline(t.avg_fold, color="#4C78A8", ls=":", lw=1)
    a.text(t.sel_fold - 0.004, t.sel_pooled, f"selected: {t.sel_pooled:.3f}", color=GOLD, ha="right", va="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.5))
    a.text(p.fold.min(), t.avg_pooled - 0.004, f"average: {t.avg_pooled:.3f}", color="#4C78A8", va="top", fontsize=8.5)
    a.set_xlabel("Mean per-fold R² (selection criterion)"); a.set_ylabel("Pooled out-of-fold R²"); a.legend(loc="upper left", frameon=False, fontsize=8.5)
    a.set_title("(a)", loc="left", fontweight="bold", x=-0.15)
    names = {labs[0]: f"Kumar et al.\n({labs[0]} oxides)", labs[1]: f"Curated\n({labs[1]})", labs[2]: f"Curated +\nexpanded ({labs[2]})"}
    rng = np.random.default_rng(0)
    for k, lab in enumerate(labs):
        p = P[(P.dataset == lab) & (P.model == "CatBoost")]; t = T[(T.dataset == lab) & (T.model == "CatBoost")].iloc[0]
        b.scatter(k + rng.uniform(-0.12, 0.12, len(p)), p.pooled, s=18, facecolor="white", edgecolor=C[k], lw=1.0, zorder=3)
        b.errorbar(k + 0.3, t.avg_pooled, yerr=t.avg_pooled_sd, fmt="o", color=C[k], capsize=3, zorder=3)
        b.text(k + 0.38, t.avg_pooled, f"{t.avg_pooled:.3f}", color=C[k], va="center", fontsize=8.5)
        b.scatter([k], [t.sel_pooled], marker="*", s=150, color=GOLD, zorder=4); b.text(k, t.sel_pooled + 0.012, f"{t.sel_pooled:.3f}", color=GOLD, ha="center", va="bottom", fontsize=8.5)
    b.set_xticks(range(3)); b.set_xticklabels([names[l] for l in labs]); b.set_xlim(-0.5, 2.8); b.set_ylabel("Pooled out-of-fold R²")
    b.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", color="grey", label="Individual partitions"),
                      Line2D([], [], marker="o", color="grey", label="Average ± SD"), Line2D([], [], marker="*", ls="", ms=11, color=GOLD, label="Selected partition")], loc="lower left", frameon=False, fontsize=8.5)
    b.set_title("(b)", loc="left", fontweight="bold", x=-0.15)
    for ax in (a, b):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "Figure_5_partition_selection")


# ------------------------------------------------------------------ Figure 6 (benchmark)
def fig6():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    B = pd.read_csv(os.path.join(R, "bench_summary.csv")); B = B[B.model.isin(["RandomForest", "SVR", "CatBoost", "XGBoost", "GPR", "LightGBM", "Ridge", "Constant"])]
    order = B[B.scheme == "comp"].sort_values("mean_rank").model.tolist()
    order = [m for m in order if m not in ("Ridge", "Constant")] + ["Ridge", "Constant"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.4), sharey=True)
    yy = np.arange(len(order))[::-1]
    for ax, (sk, name) in zip(axs, [("comp", "Unseen\ncompositions"), ("sg", "Unseen\nspace groups"), ("site", "Unseen Cr³⁺\nsubstitution sites")]):
        t = B[B.scheme == sk].set_index("model"); ridge = t.loc["Ridge", "MAE"]
        ax.axvline(ridge, color="#7f7f7f", ls="--", lw=0.9, zorder=1)
        for yi, m in zip(yy, order):
            r = t.loc[m]; sd = r.MAE_sd if sk != "site" else 0
            sg = r.get("sig", "") if isinstance(r.get("sig", ""), str) else ""
            if m == "Ridge": c, mf = "#4d4d4d", "#4d4d4d"
            elif m == "Constant": c, mf = "#b0b0b0", "#b0b0b0"
            elif sg == "▲": c, mf = "#0072BD", "#0072BD"
            elif sg == "▼": c, mf = "#D95319", "#D95319"
            else: c, mf = "#0072BD", "white"
            ax.errorbar(r.MAE, yi, xerr=sd if sd > 0 else None, fmt="o", ms=6, color=c, mfc=mf, mec=c, mew=1.3, elinewidth=1, capsize=2, zorder=3)
            ax.text(0.295 if m != "Constant" else r.MAE - 0.004, yi, f"{r.MAE:.3f}", va="center", ha="left" if m != "Constant" else "right", fontsize=7.8, color="#333333")
        ax.set_title(name, fontsize=9.6); ax.set_xlim(0.18, 0.315); ax.set_xticks([0.20, 0.24, 0.28]); ax.grid(axis="x", color="#E6E6E6", lw=0.7, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel("MAE")
    axs[0].set_yticks(yy); axs[0].set_yticklabels([NAMES.get(m, m) for m in order])
    h = [Line2D([], [], marker="o", ls="", color="#0072BD", mfc="#0072BD", label="better than Ridge"), Line2D([], [], marker="o", ls="", color="#0072BD", mfc="white", label="not significant"),
         Line2D([], [], marker="o", ls="", color="#D95319", label="worse than Ridge"), Line2D([], [], color="#7f7f7f", ls="--", label="Ridge MAE")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.55, -0.01)); fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Figure_6_benchmark")


# ------------------------------------------------------------------ Figure 7 (parity)
def fig7():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    COL = {"WCF": "#2F6EBA", "ICF": "#E3692B", "SCF": "#1A9E77"}; reg = regime(y)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 7.6), gridspec_kw={"hspace": 0.35, "wspace": 0.3})
    for row, m in enumerate(["CatBoost", "RandomForest"]):
        P = pd.read_csv(os.path.join(R, f"parity_{m}.csv"))
        for col, (key, ttl) in enumerate([("train_fit", "training"), ("pred_mean", "host-grouped out-of-fold")]):
            ax = axs[row, col]; p = P[key].values
            for lo, hi in [(1.0, ICF[0]), (ICF[0], ICF[1]), (ICF[1], 3.6)]:
                ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, color="#ECEBE7", zorder=0))
            for v in ICF: ax.axvline(v, color="#999999", ls=":", lw=0.8); ax.axhline(v, color="#999999", ls=":", lw=0.8)
            if key == "pred_mean":
                for r in ("WCF", "ICF", "SCF"):
                    mm = reg == r; ax.errorbar(y[mm], p[mm], yerr=P.pred_sd.values[mm], fmt="none", ecolor=COL[r], alpha=0.4, lw=0.8, zorder=2)
            for r in ("WCF", "ICF", "SCF"):
                mm = reg == r; ax.scatter(y[mm], p[mm], s=30, color=COL[r], edgecolor="white", lw=0.5, zorder=3)
            ax.plot([1, 3.6], [1, 3.6], "k--", lw=1.2)
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(); mae = np.mean(np.abs(y - p)); acc = np.mean(regime(p) == reg)
            ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRegime accuracy = {100 * acc:.0f} %", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB"))
            for x, t in [(1.6, "WCF"), (2.325, "ICF"), (3.0, "SCF")]: ax.text(x, 1.07, t, ha="center", fontsize=8.5, color="#555555")
            ax.set_xlim(1, 3.6); ax.set_ylim(1, 3.6); ax.set_xlabel("Experimental Dq/B"); ax.set_ylabel("Predicted Dq/B")
            ax.set_title(f"({'abcd'[2 * row + col]}) {NAMES.get(m, m)}: {ttl}", loc="left", fontsize=10.5)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
    h = [Line2D([], [], marker="o", ls="", color=COL[r], label=f"{r} (experimental)") for r in ("WCF", "ICF", "SCF")] + \
        [Line2D([], [], color="k", ls="--", label="Ideal (y = x)"), plt.Rectangle((0, 0), 1, 1, color="#ECEBE7", label="Correct regime")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0)); fig.subplots_adjust(bottom=0.12)
    save(fig, "Figure_7_parity_CatBoost_RandomForest")


# ------------------------------------------------------------------ Figure 8 (learning curves)
def fig8():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    L = pd.read_csv(os.path.join(R, "lc_summary.csv"))
    ST = {"CatBoost": ("#2F6EBA", "o", "CatBoost (original settings)"), "RandomForest": ("#E3692B", "s", "Random Forest"), "SVR": ("#E5A50A", "D", "SVR"), "Ridge": ("#DD6FA6", "v", "Ridge")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"wspace": 0.3})
    for m, (c, mk, lab) in ST.items():
        t = L[L.model == m].sort_values("n_train")
        a.plot(t.n_train, t.R2, marker=mk, color=c, label=lab, ms=5); a.fill_between(t.n_train, t.R2 - t.R2_sd, t.R2 + t.R2_sd, color=c, alpha=0.15, lw=0)
        b.plot(t.n_train, t.MAE, marker=mk, color=c, ms=5); b.fill_between(t.n_train, t.MAE - t.MAE_sd, t.MAE + t.MAE_sd, color=c, alpha=0.15, lw=0)
    t = L[L.model == "Constant"].sort_values("n_train"); b.plot(t.n_train, t.MAE, ls="--", color="grey", label="Constant (training mean)")
    a.set_ylabel("Test R² (out-of-fold)"); b.set_ylabel("Test MAE")
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        ax.set_xlabel("Training-set size (entries per fold)"); ax.set_title(lab, loc="left"); ax.grid(axis="y", color="#EEEEEE")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.31, wspace=0.3)
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, "Figure_8_learning_curves")


# ------------------------------------------------------------------ Figure 9 (b, c)
def fig9():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    T = pd.read_csv(os.path.join(R, "transfer_summary.csv")); PH = pd.read_csv(os.path.join(R, "transfer_perhost.csv"))
    ST = {"CatBoost": ("#2F6EBA", "o"), "XGBoost": ("#7FA7D9", "^"), "LightGBM": ("#5B9E5B", "+"), "RandomForest": ("#E3692B", "s"), "SVR": ("#E5A50A", "D"),
          "GPR": ("#8E6BBF", "x"), "Ridge": ("#DD6FA6", "v")}
    xs = sorted(T.stage.unique()); nmov = {1: 0, 2: int(T[T.stage == 2].n_train.iloc[0] - n_cur), 3: int(T[T.stage == 3].n_train.iloc[0] - n_cur)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"wspace": 0.3})
    for m, (c, mk) in ST.items():
        t = T[T.model == m].sort_values("stage")
        a.errorbar([nmov[s] for s in t.stage], t.MAE, yerr=t.MAE_sem, marker=mk, color=c, capsize=3, label=NAMES.get(m, m), ms=6)
    t = T[T.model == "Constant"].sort_values("stage"); a.plot([nmov[s] for s in t.stage], t.MAE, ls="--", color="#C0392B", label="Constant")
    a.set_xticks([nmov[s] for s in xs]); a.set_xlabel("Added hosts in training"); a.set_ylabel("MAE on held-out added hosts"); a.set_title("(a)", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8.5)
    inhm = d.Subset.eq("Kumar").values
    from p1lib import make
    fit = make("RandomForest", feats, seed=0).fit(d[inhm], y[inhm]).predict(d[inhm])
    b.scatter(y[inhm], fit, s=8, color="#BBBBBB", label="Training fit (inherited)", zorder=2)
    s1 = PH[(PH.stage == 1) & (PH.model == "RandomForest")].groupby("Formula").agg(y=("y", "first"), p=("pred", "mean"))
    s3 = PH[(PH.stage == 3) & (PH.model == "RandomForest")].groupby("Formula").agg(y=("y", "first"), p=("pred", "mean"))
    j = s1.join(s3, rsuffix="3", how="inner")
    for _, r in j.iterrows(): b.plot([r.y, r.y], [r.p, r.p3], color="#9DB4D3", lw=0.8, zorder=2)
    b.scatter(s1.y, s1.p, s=40, facecolor="white", edgecolor="#9DB4D3", lw=1.4, label="Added hosts, stage 1", zorder=3)
    b.scatter(s3.y, s3.p, s=36, color="#1B2F4E", edgecolor="white", lw=0.5, label="Added hosts, stage 3", zorder=4)
    for v in ICF: b.axvline(v, color="#AAAAAA", ls=":", lw=0.8); b.axhline(v, color="#AAAAAA", ls=":", lw=0.8)
    b.plot([1, 3.6], [1, 3.6], "k--", lw=1.1); b.set_xlim(1, 3.6); b.set_ylim(1, 3.6)
    b.set_xlabel("Experimental Dq/B"); b.set_ylabel("Predicted Dq/B (Random Forest)"); b.set_title("(b)", loc="left"); b.legend(loc="lower right", frameon=False, fontsize=8.5)
    for ax in (a, b):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "Figure_9bc_transfer_to_added_hosts")


# ------------------------------------------------------------------ Figure 3 (SHAP, three stages)
LABELS = {"avg_Mulliken EN": "Avg. Mulliken EN", "avg_First ionization energy (kJ/mol)": "Avg. 1st ionization energy", "1/r2": "1/Δr²", "avg_Metallic valence": "Avg. metallic valence",
          "avg_Martynov-Batsanov EN": "Avg. Martynov–Batsanov EN", "beta": "β angle", "SGR No.": "Space-group number", "avg_Number of outer shell electrons": "Avg. outer-shell electrons",
          "X": "Cr³⁺ concentration x", "max_metal_ligand_bond_length": "Max. metal–ligand bond length", "std_Mendeleev number": "Std. Mendeleev number", "volume_per_atom": "Volume per atom",
          "volume_per_fu": "Volume per f.u.", "polyhedron volume": "Polyhedron volume"}


def fig3():
    EB = ["avg_Mulliken EN", "avg_First ionization energy (kJ/mol)", "avg_Martynov-Batsanov EN"]; LS = ["max_metal_ligand_bond_length", "polyhedron volume"]
    Sh = pd.read_csv(os.path.join(R, "shap_shares.csv")); labs = list(dict.fromkeys(Sh.dataset))
    sty = [(labs[0], f"Kumar et al., as published ({labs[0]})", "#9DB4D3", "o", True), (labs[1], f"Kumar et al., curated ({labs[1]})", "#4C78A8", "^", False),
           (labs[2], f"Final dataset ({labs[2]})", "#1B2F4E", "o", False)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 4.7), gridspec_kw={"width_ratios": [1.5, 1.2]})
    fin = Sh[(Sh.dataset == labs[2]) & ~Sh.descriptor.str.startswith("GROUP")].set_index("descriptor")
    order = fin.share.sort_values().index.tolist(); yy = np.arange(len(order))
    for i, nme in enumerate(order):
        if nme in EB: a.axhspan(i - 0.5, i + 0.5, color="#DCEFE3", zorder=0)
        if nme in LS: a.axhspan(i - 0.5, i + 0.5, color="#F3E6C8", zorder=0)
    for k, (lab, name, c, mk, hollow) in enumerate(sty):
        t = Sh[(Sh.dataset == lab)].set_index("descriptor")
        a.errorbar(t.loc[order, "share"], yy + [-0.22, 0, 0.22][k], xerr=t.loc[order, "sd"], fmt=mk, ms=5.5, color=c, mfc="white" if hollow else c, mec=c, mew=1.3, elinewidth=0.9, capsize=1.8, label=name, zorder=3)
    a.set_yticks(yy); a.set_yticklabels([LABELS[x] for x in order], fontsize=9.5); a.set_xlabel("Share of total mean |SHAP| (%)"); a.set_xlim(0, 15)
    a.grid(axis="x", color="#E6E6E6", lw=0.7, zorder=0); a.set_ylim(-0.6, len(order) - 0.4); a.set_title("(a)", loc="left", fontweight="bold")
    G = [("GROUP: electron-binding", "Electron\nbinding", "#2E7D4F"), ("GROUP: local-structure", "Local\nstructure", "#B8860B"), ("GROUP: other", "Other", "#8C8C8C")]
    w = 0.28
    for k, (lab, name, c, mk, hollow) in enumerate(sty):
        t = Sh[Sh.dataset == lab].set_index("descriptor"); vals = [(t.loc[g, "share"], t.loc[g, "sd"]) for g, _, _ in G]; xs = np.arange(3) + (k - 1) * w
        b.bar(xs, [v[0] for v in vals], w * 0.92, yerr=[v[1] for v in vals], color=c if not hollow else "white", edgecolor=c, linewidth=1.3, capsize=2, zorder=3)
        for x, v in zip(xs, vals): b.text(x, v[0] + v[1] + 1.0, f"{v[0]:.0f}", ha="center", fontsize=7.4)
    b.set_xticks(range(3)); b.set_xticklabels([g[1] for g in G], fontsize=8.0, linespacing=1.0)
    for tl, (_, _, c) in zip(b.get_xticklabels(), G): tl.set_color(c)
    b.set_ylabel("Summed share (%)"); b.set_ylim(0, 72); b.grid(axis="y", color="#E6E6E6", lw=0.7, zorder=0); b.set_title("(b)", loc="left", fontweight="bold")
    for ax in (a, b):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    h, l = a.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "Figure_3_SHAP_three_dataset_stages")


# ------------------------------------------------------------------ Figure 10 (SHAP, final)
def lowess(x, yv, frac=0.35):
    o = np.argsort(x); x, yv = x[o], yv[o]; nn = max(5, int(frac * len(x))); out = np.empty_like(yv)
    for i, x0 in enumerate(x):
        dd = np.abs(x - x0); h = np.sort(dd)[nn - 1] or 1e-9; w = np.clip(1 - (dd / h) ** 3, 0, None) ** 3
        A = np.vstack([np.ones_like(x), x - x0]).T; W = np.diag(w); beta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ yv, rcond=None)[0]; out[i] = beta[0]
    return x, out


def fig10():
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fe = pd.read_csv(os.path.join(R, "shap_features.csv")).iloc[:, 0].tolist(); lab = str(n)
    cmap = LinearSegmentedColormap.from_list("bg", ["#0E5E4A", "#5DB89A", "#F2EAD0", "#D9A441", "#7A4F12"])
    fig = plt.figure(figsize=(7.6, 8.4))
    top = fig.add_gridspec(1, 2, left=0.27, right=0.86, top=0.97, bottom=0.6, wspace=1.15)
    gs = fig.add_gridspec(2, 3, left=0.1, right=0.97, top=0.5, bottom=0.06, hspace=0.6, wspace=0.42)
    for k, m in enumerate(["CatBoost", "RandomForest"]):
        sv = np.load(os.path.join(R, f"shap_oof_{lab}_{m}.npy")); ax = fig.add_subplot(top[0, k])
        imp = np.abs(sv).mean(0); order = np.argsort(imp)
        for i, j in enumerate(order):
            v = d[fe[j]].values.astype(float); pc = pd.Series(v).rank(pct=True).values
            sc = ax.scatter(sv[:, j], i + np.random.default_rng(j).uniform(-0.28, 0.28, n), c=pc, cmap=cmap, s=5, vmin=0, vmax=1, lw=0)
        ax.axvline(0, color="#777777", lw=0.8); ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[fe[j]] for j in order], fontsize=7.8)
        ax.set_xlabel("SHAP value (impact on predicted Dq/B)", fontsize=8.3); ax.set_title(f"({'ab'[k]}) {NAMES.get(m, m)}", fontsize=10)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    cax = fig.add_axes([0.91, 0.62, 0.012, 0.33]); cb = fig.colorbar(sc, cax=cax); cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.set_label("Descriptor value (percentile)", fontsize=8.5)
    for k, m in enumerate(["CatBoost", "RandomForest"]):
        sv = np.load(os.path.join(R, f"shap_oof_{lab}_{m}.npy"))
        for c, (fname, xl) in enumerate([("avg_Mulliken EN", "Avg. Mulliken EN (eV)"), ("max_metal_ligand_bond_length", "Max. metal–ligand bond length (Å)"), ("polyhedron volume", "Polyhedron volume (Å³)")]):
            ax = fig.add_subplot(gs[k, c]); j = fe.index(fname); x = d[fname].values.astype(float)
            ax.scatter(x, sv[:, j], s=7, color="#3E9C7E", alpha=0.75, lw=0); xs, ys = lowess(x, sv[:, j]); ax.plot(xs, ys, color="#D4A017", lw=1.8)
            ax.axhline(0, color="#999999", ls="--", lw=0.8); ax.set_xlabel(xl, fontsize=8.5)
            if c == 0: ax.set_ylabel("SHAP value"); ax.set_title(f"({'cd'[k]}) {NAMES.get(m, m)}", loc="left", fontsize=10)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "Figure_10_SHAP_out_of_fold")


if __name__ == "__main__":
    only = os.environ.get("ONLY", "").split(",")
    for f in (fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        if only != [""] and f.__name__ not in only: continue
        f(); print(f.__name__, "ok", flush=True)
