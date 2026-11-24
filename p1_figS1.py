"""Supplementary Figure S1: Pearson correlation matrices of the 14 descriptors at three dataset stages.
Usage: python p1_figS1.py <published159.xlsx> <curated.xlsx> <final.xlsx> <out_dir>"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ORDER = [("avg_Mulliken EN", "Avg. Mulliken EN"), ("avg_First ionization energy (kJ/mol)", "Avg. 1st ionization energy"),
         ("1/r2", "1/Δr²"), ("avg_Metallic valence", "Avg. metallic valence"), ("avg_Martynov-Batsanov EN", "Avg. Martynov–Batsanov EN"),
         ("beta", "β angle"), ("SGR No.", "Space group no."), ("avg_Number of outer shell electrons", "Avg. outer-shell electrons"),
         ("X", "Cr³⁺ concentration x"), ("max_metal_ligand_bond_length", "Max. M–O bond length"),
         ("std_Mendeleev number", "SD Mendeleev number"), ("volume_per_atom", "Volume per atom"), ("volume_per_fu", "Volume per f.u."),
         ("polyhedron volume", "Polyhedron volume")]
CMAP = LinearSegmentedColormap.from_list("gw", ["#2E7D4F", "#A9CDB4", "#F4F4F2", "#E2CB94", "#B8860B"])

if __name__ == "__main__":
    p159, pcur, pfin, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    sets = [(pd.read_excel(p159), "(a) Inherited oxides as published"), (pd.read_excel(pcur), "(b) Curated inherited subset"),
            (pd.read_excel(pfin), "(c) Final dataset")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.9), gridspec_kw={"wspace": 0.55})
    cols = [c for c, _ in ORDER]; labs = [l for _, l in ORDER]; n = len(cols)
    for ax, (d, title) in zip(axs, sets):
        C = d[cols].astype(float).corr().values
        im = ax.imshow(C, cmap=CMAP, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                v = C[i, j]; strong = abs(v) > 0.8
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0, fontweight="bold" if strong else "normal",
                        color="white" if strong else "#222222")
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right"); ax.set_yticklabels(labs)
        ax.set_xticks(np.arange(-0.5, n), minor=True); ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{title} ({len(d)})", fontweight="bold", fontsize=10)
        iu = np.triu_indices(n, 1); print(title, len(d), "pairs |r|>0.8:", int((np.abs(C[iu]) > 0.8).sum()),
                                          "max |r| with x:", round(np.abs(np.delete(C[cols.index('X')], cols.index('X'))).max(), 2))
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.02); cb.set_label("Pearson correlation coefficient r")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"FS1_Pearson_correlation.{ext}"), dpi=500 if ext == "png" else None, bbox_inches="tight")
    print("saved")
