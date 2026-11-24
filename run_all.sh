#!/bin/sh
# Reproduces results_183/ and the figures of the final version (about 5 min on 8 cores).
set -e
D=data; R=results_183; mkdir -p $R figures
python p1_bench.py $D/train183.xlsx $R
python p1_partition.py $R 159=$D/Kumar_original_oxides_159.xlsx:CatBoost,Ridge,Constant \
    133=$D/curated_oxides_133.xlsx:CatBoost,Ridge,Constant \
    183=$D/train183.xlsx:CatBoost,XGBoost,LightGBM,RandomForest,SVR,Ridge,Constant
python p1_transfer.py $D/train183.xlsx $R 26 9
python p1_shap.py $R 159=$D/Kumar_original_oxides_159.xlsx 133=$D/curated_oxides_133.xlsx 183=$D/train183.xlsx --fig10 183
python p1_lc_nested.py $D/train183.xlsx $R
python p1_stats.py $D/train183.xlsx $D/curated_oxides_133.xlsx $R $R
python p1_figures.py $R $D/train183.xlsx $D/curated_oxides_133.xlsx figures
python p1_figS1.py $D/Kumar_original_oxides_159.xlsx $D/curated_oxides_133.xlsx $D/train183.xlsx figures
python p1_family.py $D/train183.xlsx $R          # Table S11 (family-grouped transfer), about 10 min on 2 cores
