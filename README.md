# Code for the Dq/B benchmark of Cr³⁺-doped oxide hosts

Re-implementation of all analyses of *Data Curation and Generalization in Machine-Learning Prediction of Dq/B for
Cr³⁺-Doped Oxide Phosphors* (Đurković, Ćirić, Ristić, Dramićanin). The model definitions (`common185.py`) are the
ones of the MATLAB application; the file name is kept for compatibility.

## Data

The data are deposited on Zenodo (DOI https://doi.org/10.5281/zenodo.24508999) and are not part of this repository. The record contains `train183.xlsx` (final dataset: 183 entries, 133 inherited and 50 added, 181 compositions, 14 descriptors) and the output files of all analyses. Place `train183.xlsx` in `data/` and all other files of the record (except `README_DATA.md`) in `results_183/` of this folder.

The oxide entries of Kumar et al. as published, the 33 non-oxide entries of Table S1 and the separate file of the curated inherited subset are not redistributed. The inherited entries are contained in `train183.xlsx` (column `Subset` = `Kumar`). The scripts `p1_partition.py`, `p1_stats.py` and `p1_figS1.py` additionally read the original compilation, which is available from the DqBpredictor repository of Kumar et al. (https://github.com/BrgochGroup/DqBpredictor, Chem. Mater. 37, 7762 (2025)).

## Scripts

Run from this folder. XGBoost and LightGBM need OpenMP (on macOS: `libomp`).

| Script | Output (in the results folder) | Paper |
|---|---|---|
| `p1_bench.py <data.xlsx> <out>` | `bench_summary.csv`, `bench_oof.npz` | Fig. 6, Table S10, Table S8 (i, ii) |
| `p1_partition.py <out> <label>=<data.xlsx>:<models> ...` | `partition_summary.csv`, `partition_per_state.csv` | Table 4, Fig. 5, Table S9 |
| `p1_transfer.py <data.xlsx> <out> [26 9]` | `transfer_summary.csv`, `transfer_perhost.csv` | Fig. 9 |
| `p1_shap.py <out> 159=<..> 133=<..> 183=<..> --fig10 183` | `shap_shares.csv`, `shap_oof_*.npy` | Fig. 3, Fig. 10 |
| `p1_family.py <data.xlsx> <out>` | `family_transfer_summary.csv`, `family_transfer_perhost.csv`, `family_definition.csv` | Table S11, Sec. III C |
| `p1_lc_nested.py <data.xlsx> <out>` | `lc_summary.csv`, `nested.csv` | Fig. 8, Table S8 (iii, iv) |
| `p1_stats.py <data.xlsx> <curated.xlsx> <bench_dir> <out>` | `stats.json`, `parity_*.csv` | Table 3, Fig. 7, Sections 3.1–3.2 |
| `p1_figures.py <res_dir> <final.xlsx> <curated.xlsx> <fig_dir>` | Figures 2–10 (png, pdf) | |
| `p1_figS1.py <159.xlsx> <curated.xlsx> <final.xlsx> <out>` | Figure S1 | |
| `figure1/` | Figure 1 (exact d³ Tanabe–Sugano diagram, C/B = 4.5) and Table S12 (`sweep.py`) | |

`run_all.sh` lists the exact command lines used for `results_183/` and the figures.

## Results

The results of the final version (183 entries) are in the Zenodo record. Copy them into `results_183/` to make the figures without recomputing, or run `run_all.sh` to recompute everything.

Environment used for `results_183/`: Python 3.14, scikit-learn 1.9.1, CatBoost 1.2.10, XGBoost 3.4.1,
LightGBM 4.7.0, SHAP 0.54.1, NumPy 2.5.5, pandas 3.1.1 (macOS, CPU).

## Use of AI tools

Parts of this code (the Python re-implementation and the MATLAB application) were written with the assistance of a large language model (Claude, Anthropic). All code was reviewed and run by the authors; the grouped benchmark (Table S10) was reproduced independently with the unmodified code.

## Citation

Please cite the data record (DOI https://doi.org/10.5281/zenodo.24508999) and the article (Đurković, Ćirić, Ristić, Dramićanin; reference to be added upon publication).

## Licence

Copyright (C) 2026 Snežana Đurković, Aleksandar Ćirić, Zoran Ristić, Miroslav D. Dramićanin. The software is released under the GNU General Public License v3.0 (see `LICENSE`). The data record on Zenodo is licensed under CC BY 4.0.
