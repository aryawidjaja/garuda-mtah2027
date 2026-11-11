"""Realism check: Metrica sample games have real Metrica-style tracking (ball missing mostly in
stoppages, no simulation needed) and event-derived PROXY labels (dead time under-counted).

  A  train on the 7 degraded DFL matches only -> test each Metrica game as delivered
  B  A + the other two Metrica games (as delivered, proxy labels) -> test the third

    python garuda/scripts/experiment_metrica.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import experiment as ex  # noqa: E402
from garuda import decode, ext_data, features  # noqa: E402


def raw_features(mid):
    """No degradation: the source already has realistic gaps. Rows: plain view then swapped."""
    d = ext_data.load_match(mid)
    X0, X1, ys, pers, n25 = [], [], [], [], []
    for k, sub in enumerate(ex.periods(d)):
        _, a = features.build(sub)
        _, b = features.build(sub, swap=True)
        y = features.pool_labels(sub["label"].astype(np.int8))
        X0.append(a); X1.append(b); ys.append(y); pers.append(np.full(len(y), k)); n25.append(len(sub["label"]))
    return np.concatenate(X0), np.concatenate(X1), np.concatenate(ys), np.concatenate(pers), n25


def fit(Xs, ys):
    import lightgbm as lgb
    X, y = np.concatenate(Xs), np.concatenate(ys)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             min_child_samples=200, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.6, reg_lambda=1.0, n_jobs=4, verbose=-1,
                             random_state=0, deterministic=True)
    return clf.fit(X, y), np.bincount(y, minlength=3) / len(y)


def evaluate(clf, prior, mid, h):
    X0, X1, _, per, n25 = raw_features(mid)
    p = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
    gts = [sub["label"].astype(np.int8) for sub in ex.periods(ext_data.load_match(mid))]
    vit, mbr = [], []
    for k in range(len(n25)):
        e = decode.emissions(p[per == k], prior, 1.0)
        v = h.viterbi(e)
        s = h.sample_paths(e, 24, seed=k)
        c = [v] + list(s)
        vit.append(ex.to25(v, n25[k]))
        mbr.append(ex.to25(c[decode.mbr_select(c, s)[0]], n25[k]))
    return {"argmax": ex.score(gts, [ex.to25(p[per == k].argmax(1), n25[k]) for k in range(len(n25))]),
            "viterbi": ex.score(gts, vit), "mbr": ex.score(gts, mbr)}


if __name__ == "__main__":
    dfl = ext_data.list_matches("idsse")
    met = ext_data.list_matches("metrica")
    Xd, yd = [], []
    for i, m in enumerate(dfl):
        for k in range(ex.K_DRAWS):
            _, X, y, _, _, _ = ex.match_features(m, 100 * i + k, True)
            Xd.append(X); yd.append(y)
    h, _ = ex.hsmm_from_labels(dfl, (1, 1, 1), 1.0)
    clfA, priorA = fit(Xd, yd)
    raw = {m: raw_features(m) for m in met}
    res = {}
    for m in met:
        res[f"A_{m}"] = evaluate(clfA, priorA, m, h)
        others = [o for o in met if o != m]
        Xo = Xd + [np.concatenate([raw[o][0], raw[o][1]]) for o in others]
        yo = yd + [np.concatenate([raw[o][2], np.where(raw[o][2] == 2, 2, 1 - raw[o][2])]) for o in others]
        clfB, priorB = fit(Xo, yo)
        res[f"B_{m}"] = evaluate(clfB, priorB, m, h)
        print(m, {k: {kk: round(v["spell_f1_mean"], 4) for kk, v in res[k].items()}
                  for k in (f"A_{m}", f"B_{m}")}, flush=True)
    json.dump(res, open(os.path.join(ex.OUT, f"metrica_{ex.TARGET}.json"), "w"), indent=1)
    for setup in "AB":
        for dec in ("argmax", "viterbi", "mbr"):
            v = [res[f"{setup}_{m}"][dec]["spell_f1_mean"] for m in met]
            mf = [res[f"{setup}_{m}"][dec]["macro_f1"] for m in met]
            print(f"{setup} {dec:8s} spellF1 {np.mean(v):.4f}  macroF1 {np.mean(mf):.4f}")
