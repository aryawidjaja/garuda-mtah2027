"""Training-mix helpers for train_final.py: degraded Metrica draws and the weighted LightGBM fit."""
import os, sys
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment as ex  # noqa: E402
from garuda import degrade, ext_data, features  # noqa: E402

features.VERSION = 1


def met_degraded(mid, seed):
    """One degraded draw of a Metrica game: (X plain, X swapped, y, per, n25); cached."""
    path = os.path.join(ex.CACHE, f"{mid}_deg_s{seed}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return z["X0"], z["X1"], z["y"], z["per"], list(z["n25"])
    rng = np.random.default_rng(seed)
    X0, X1, ys, pers, n25 = [], [], [], [], []
    for k, sub in enumerate(ex.periods(ext_data.load_match(mid))):
        dd = degrade.degrade(sub, sub["label"], degrade.sample_params(rng), rng)
        X0.append(features.build(dd)[1]); X1.append(features.build(dd, swap=True)[1])
        y = features.pool_labels(sub["label"].astype(np.int8))
        ys.append(y); pers.append(np.full(len(y), k)); n25.append(len(sub["label"]))
    out = (np.concatenate(X0), np.concatenate(X1), np.concatenate(ys), np.concatenate(pers), n25)
    np.savez_compressed(path, X0=out[0], X1=out[1], y=out[2], per=out[3], n25=np.array(n25))
    return out


def both_views(X0, X1, y):
    return [X0, X1], [y, np.where(y == 2, 2, 1 - y)]


def fit(Xs, ys, ws):
    X, y = np.concatenate(Xs), np.concatenate(ys)
    w = np.concatenate([np.full(len(a), wt, np.float32) for a, wt in zip(Xs, ws)])
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, n_jobs=4,
                             verbose=-1, random_state=0, deterministic=True)
    clf.fit(X, y, sample_weight=w)
    return clf, np.bincount(y, weights=w, minlength=3) / w.sum()

