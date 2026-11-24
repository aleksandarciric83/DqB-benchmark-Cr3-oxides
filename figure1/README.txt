Code for the figures changed in v6

ts_d3.py, sweep.py  Exact Tanabe-Sugano energies of octahedral d3 (Cr3+): full 120-determinant
                    basis, Slater-Condon parameters from B and C (B = F2 - 5F4, C = 35F4),
                    octahedral crystal field. Checks: 4P-4F = 15B, 2G-4F = 4B+3C, eq. (2) returns
                    the input Dq/B. Crossing 2E/4T2: C/B = 4.0 -> 1.92, 4.5 -> 2.07, 5.25 -> 2.31.
                    Run: python3 sweep.py 4.5        (needs numpy, sympy)
fig1a.py            Figure 1(a) with C/B = 4.5; panel (b) is taken unchanged from the previous figure.
fig3_v6.py          Figure 3 from the per-fold SHAP files (10x10 CV, CatBoost, Kumar hyperparameters);
                    final dataset recomputed with Dq/B(CdWO4) = 1.818.
fig_bench.py        Figure 6 (former Table 5) from Table5_benchmark_CdWO4_1.818.xlsx.
fig1a.py: python3 fig1a.py <previous Figure 1 png> (panel b is copied from it).
fig3_v6.py and fig_bench.py are kept for reference only; the figures of the final version are made by p1_figures.py.
