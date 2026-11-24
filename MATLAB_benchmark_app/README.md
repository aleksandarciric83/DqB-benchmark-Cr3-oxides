# Cr³⁺ Dq/B benchmark – MATLAB app

GUI for rerunning the benchmark of *Data Curation and Generalization in Machine-Learning Prediction of Dq/B for Cr³⁺-Doped Oxide Phosphors* (Đurković, Ćirić, Ristić, Dramićanin).

## Start

```matlab
cd Cr3DqB_Benchmark_App
app = Cr3DqBBenchmarkApp;
```

The final dataset (`Code/data/train183.xlsx` of the package, 183 entries, 14 descriptors) is loaded automatically. Any other xlsx/csv file with a `Dq/B` column can be loaded. The optional columns `Formula`, `SGR No.`, `Cr_site` and `Subset` (Kumar/Added) are needed for the grouped schemes and the transfer test.

Requirements: MATLAB R2020b or newer, Statistics and Machine Learning Toolbox (Ridge and Constant work without it).

## What can be set

| Control | Options |
|---|---|
| Engine | **MATLAB**: native models (boosted trees/LSBoost, Random Forest, SVR, GPR, ridge, constant). **Python**: the code of the paper (`Code/common185.py` of the package): CatBoost (settings of Kumar et al.), XGBoost, LightGBM, Random Forest, SVR, GPR, ridge, constant |
| Validation | Unseen compositions (formula-grouped k-fold) · Unseen space groups · Unseen substitution sites (leave-one-cation-out) · Ungrouped k-fold (as Kumar et al.) · Transfer inherited → added hosts · Partition selection (original procedure, 31 partitions) |
| Folds, partitions, seed | k-fold settings; partition r uses seed + r − 1 |
| Models, descriptors | any subset |

## Output

- **Results**: pooled R², MAE, RMSE (mean ± SD over partitions); ΔMAE against ridge regression with a 95 % bootstrap interval over formulas; crystal-field-regime accuracy and number of ICF hosts assigned correctly. For partition selection: selected vs averaged R².
- **Parity plot**: partition-averaged out-of-fold predictions with SD, coloured by regime, ICF window shaded.
- **Model comparison**: MAE bar chart with the constant predictor as reference.
- **Dataset**, **Log**, and **Export** to xlsx (settings, summary, per-entry predictions, per-partition metrics).

## Which engine to use

The **Python** engine reproduces the numbers of the paper exactly (Table S10; checked for all eight models in the three host-grouped schemes, e.g. unseen compositions Random Forest R² 0.430 / MAE 0.187, SVR 0.417 / 0.193, CatBoost 0.381 / 0.196, ridge 0.387 / 0.203; unseen substitution sites SVR 0.361 / 0.201). For the transfer test the paper averages the tree ensembles and GPR over model seeds 0–4, the app uses one seed, so CatBoost, XGBoost and Random Forest differ from Figure 9a by up to 0.04 in R². Set the Python executable (e.g. `C:\Users\...\python.exe`) and the folder `Code/` of the package once; install the packages with `pip install -r python/requirements.txt` (Python 3.13 or newer).

The **MATLAB** engine uses MATLAB's own implementations. Ridge and the constant predictor follow the same definitions as in the paper; Random Forest, SVR and GPR follow the same settings, but the implementations differ from scikit-learn, so their values differ slightly. Boosted trees (LSBoost, depth ~3, 500 trees, learning rate 0.1) stand in for CatBoost/XGBoost/LightGBM, which have no MATLAB equivalent.

The 95 % interval of ΔMAE against ridge regression uses 4000 bootstrap resamples over formulas, as in the paper. Runs are parallel when the Parallel Computing Toolbox is available; every fit sets its own seed, so parallel and serial runs give identical results.

Datasets (kept only once, in `Code/data/` of the package): `train183.xlsx` is the final dataset of the paper (183 entries; Ga₅GdO₁₂ and BaAl₂O₄ removed from the earlier 185-entry version), `curated_oxides_133.xlsx` the curated inherited subset and `Kumar_original_oxides_159.xlsx` the inherited oxide entries as published.

## Files

| File | Purpose |
|---|---|
| `Cr3DqBBenchmarkApp.m` | GUI |
| `lib/dqb_load.m` | read dataset |
| `lib/dqb_splits.m` | grouped / ungrouped partitions |
| `lib/dqb_fit_predict.m` | models (scaling and log10(1/R²) fitted on the training part only) |
| `lib/dqb_run.m`, `lib/dqb_summary.m` | cross-validation, metrics, bootstrap, regimes |
| `lib/dqb_partition.m` | original evaluation procedure |
| `lib/dqb_parmap.m`, `lib/dqb_pool.m`, `lib/dqb_fit_task.m`, `lib/dqb_cost.m` | parallel execution |
| `lib/matlab_bridge.py` | call of the Python benchmark code |
| `run_example.m` | the same without GUI |
