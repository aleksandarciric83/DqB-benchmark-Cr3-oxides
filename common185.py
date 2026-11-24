"""Shared definitions (data loading, models, validation splits, metrics) for the Dq/B benchmark of Cr3+-doped oxide hosts.
The model settings are those of the MATLAB application; the file name is kept for compatibility with earlier versions."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import xgboost as xgb, lightgbm as lgb

def load(path="data/train183.xlsx"):
    d = pd.read_excel(path)
    feats = [c for c in d.columns if c not in ("Formula", "Dq/B", "Subset", "Cr_site",
                                               "max_First ionization energy (kJ/mol)")]
    return d, feats

class Model:
    """Uniform fit/predict on a DataFrame. Scaling / transforms are fitted on the training part only."""
    def __init__(self, kind, feats, **kw): self.kind, self.feats, self.kw = kind, feats, kw
    def _arr(self, X, log_r2=False):
        A = X[self.feats].values.astype(float).copy()
        if log_r2: j = self.feats.index("1/r2"); A[:, j] = np.log10(A[:, j])
        return A
    def fit(self, X, y):
        k, kw = self.kind, dict(self.kw); seed = kw.pop("seed", 0)
        if k == "dummy":  self.m = DummyRegressor().fit(self._arr(X), y)
        elif k == "ridge": self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(self._arr(X, True), y)
        elif k == "catboost":
            self.cat = kw.pop("cat", False)
            Xc = X[self.feats].copy()
            if self.cat: Xc["SGR No."] = Xc["SGR No."].astype(int).astype(str)
            self.m = CatBoostRegressor(verbose=0, thread_count=1, allow_writing_files=False, random_seed=seed, **kw)
            self.m.fit(Xc, y, cat_features=["SGR No."] if self.cat else None)
        elif k == "xgb":  self.m = xgb.XGBRegressor(n_jobs=1, verbosity=0, random_state=seed, **kw).fit(self._arr(X), y)
        elif k == "lgbm": self.m = lgb.LGBMRegressor(n_jobs=1, verbose=-1, random_state=seed, **kw).fit(self._arr(X), y)
        elif k == "rf":   self.m = RandomForestRegressor(n_jobs=1, random_state=seed, **kw).fit(self._arr(X), y)
        elif k == "gpr":
            self.log = kw.get("log_r2", True); nf = len(self.feats)
            kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(nf), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(0.1, (1e-5, 1))
            self.m = make_pipeline(StandardScaler(), GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2, random_state=seed)).fit(self._arr(X, self.log), y)
        elif k == "svr":  # hyperparameters chosen by 5-fold grid search inside the training part
            grid = {"svr__C": [0.3, 1, 3, 10, 30, 100], "svr__gamma": ["scale", 0.003, 0.01, 0.03, 0.1], "svr__epsilon": [0.01, 0.05, 0.1]}
            self.m = GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")), grid, cv=5,
                                  scoring="neg_mean_squared_error", n_jobs=1).fit(self._arr(X, True), y)
        return self
    def predict(self, X):
        k = self.kind
        if k == "catboost":
            Xc = X[self.feats].copy()
            if self.cat: Xc["SGR No."] = Xc["SGR No."].astype(int).astype(str)
            return self.m.predict(Xc)
        if k in ("ridge", "svr"): return self.m.predict(self._arr(X, True))
        if k == "gpr": return self.m.predict(self._arr(X, self.log))
        return self.m.predict(self._arr(X))
    def predict_std(self, X):
        _, s = self.m.predict(self._arr(X, self.log), return_std=True); return s

KUMAR_CB = dict(depth=4, iterations=500, learning_rate=0.09293, l2_leaf_reg=1.9172)   # Kumar et al. settings
SPECS = {
    # main benchmark: 6 models + 2 baselines
    "CatBoost":      ("catboost", dict(KUMAR_CB)),
    "XGBoost":       ("xgb", dict(max_depth=3, n_estimators=500, learning_rate=0.1, reg_lambda=5, subsample=0.8)),
    "LightGBM":      ("lgbm", dict(max_depth=3, num_leaves=4, n_estimators=200, learning_rate=0.1)),
    "RandomForest":  ("rf", dict(n_estimators=500, max_features=1.0)),
    "SVR":           ("svr", {}),
    "GPR":           ("gpr", dict(log_r2=True)),
    "Ridge":         ("ridge", {}),
    "Constant":      ("dummy", {}),
    # sensitivity variants (SI)
    "CatBoost_catSG":("catboost", dict(KUMAR_CB, cat=True)),
    "GPR_raw1/r2":   ("gpr", dict(log_r2=False)),
}
MAIN = ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "SVR", "GPR", "Ridge", "Constant"]

def make(name, feats, seed=0):
    kind, kw = SPECS[name]; return Model(kind, feats, seed=seed, **kw)

def splits(d, scheme, rep):
    n = len(d)
    if scheme == "random":       # 10-fold, entries of the same formula kept together
        g = d.Formula.astype("category").cat.codes.values
        return list(GroupKFold(10, shuffle=True, random_state=rep).split(np.zeros(n), groups=g))
    if scheme == "space_group":
        return list(GroupKFold(10, shuffle=True, random_state=rep).split(np.zeros(n), groups=d["SGR No."].astype(int).values))
    if scheme == "cr_site":
        return list(LeaveOneGroupOut().split(np.zeros(n), groups=d.Cr_site.values))
    if scheme == "plain_kfold":  # as in Kumar et al. (no grouping)
        return list(KFold(10, shuffle=True, random_state=rep).split(np.zeros(n)))
    raise ValueError(scheme)

REPS = {"random": 10, "space_group": 5, "cr_site": 1, "plain_kfold": 10}

def metrics(y, p):
    mse = mean_squared_error(y, p)
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mse)))
