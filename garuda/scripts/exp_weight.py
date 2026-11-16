"""Does the DFL/SkillCorner bulk dilute Metrica's labelling convention? Metrica LOGO, paired per game.

  S    shipped recipe (v1 features, no TV cue): 7 DFL x 3 degraded draws + 20 SkillCorner native
       + 2 Metrica games clean, all weight 1
  W5   7 DFL x 3 draws + 2 Metrica games clean AND 3 degraded draws each (no z: no air loss or parallax),
       no SkillCorner labels, Metrica rows weighted 5
  W20  same, Metrica rows weighted 20
Each tested on the held-out Metrica game against its labels, clean as delivered AND one degraded draw
(seed fixed per game; the target-like readout). HSMM from the DFL relabel, as experiment_metrica.py.
Predeclared: adopt the smallest weight with >= +0.02 over S on the degraded readout, 2/3 games; then one
DFL dev run for it, which must not drop more than 0.01 below 0.642.

    python garuda/scripts/exp_weight.py
"""
import json, os, sys
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment as ex, experiment_metrica as em  # noqa: E402
from experiment_sources import sc_features  # noqa: E402
from garuda import decode, degrade, ext_data, features  # noqa: E402

features.VERSION = 1
DFL, MET, SC = ext_data.list_matches("idsse"), ext_data.list_matches("metrica"), ext_data.list_matches("skillcorner")
TEST_DRAW = 7_000


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


def score(clf, prior, h, X0, X1, per, n25, gts):
    p = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
    parts = [ex.to25(h.viterbi(decode.emissions(p[per == k], prior, 1.0)), n) for k, n in enumerate(n25)]
    return ex.score(gts, parts)["spell_f1_mean"]


if __name__ == "__main__":
    Xd, yd = [], []
    for i, m in enumerate(DFL):
        for k in range(ex.K_DRAWS):
            _, X, y, _, _, _ = ex.match_features(m, 100 * i + k, True)
            Xd.append(X); yd.append(y)
    h, _ = ex.hsmm_from_labels(DFL, (1, 1, 1), 1.0)
    Xs_sc, ys_sc = [], []
    for m in SC:
        a, b, y = sc_features(m)[:3]
        xx, yy = both_views(a, b, y); Xs_sc += xx; ys_sc += yy
    clean = {m: em.raw_features(m) for m in MET}
    res = {a: {"clean": [], "degraded": []} for a in ("S", "W5", "W20")}
    for g, m in enumerate(MET):
        oth = [o for o in MET if o != m]
        gts = [s["label"].astype(np.int8) for s in ex.periods(ext_data.load_match(m))]
        tc = clean[m]
        td = met_degraded(m, TEST_DRAW + g)
        mc_X, mc_y, md_X, md_y = [], [], [], []
        for o in oth:
            xx, yy = both_views(*clean[o][:3]); mc_X += xx; mc_y += yy
            for k in range(ex.K_DRAWS):
                xx, yy = both_views(*met_degraded(o, 500 + 10 * MET.index(o) + k)[:3]); md_X += xx; md_y += yy
        arms = {"S": (Xd + Xs_sc + mc_X, yd + ys_sc + mc_y, [1] * (len(Xd) + len(Xs_sc) + len(mc_X))),
                "W5": (Xd + mc_X + md_X, yd + mc_y + md_y, [1] * len(Xd) + [5] * (len(mc_X) + len(md_X))),
                "W20": (Xd + mc_X + md_X, yd + mc_y + md_y, [1] * len(Xd) + [20] * (len(mc_X) + len(md_X)))}
        for a, (Xs, ys, ws) in arms.items():
            clf, prior = fit(Xs, ys, ws)
            res[a]["clean"].append(score(clf, prior, h, tc[0], tc[1], tc[3], tc[4], gts))
            res[a]["degraded"].append(score(clf, prior, h, td[0], td[1], td[3], td[4], gts))
            print(m, a, {k: round(v[-1], 3) for k, v in res[a].items()}, flush=True)

    for r in ("degraded", "clean"):
        b = np.array(res["S"][r])
        print(f"\nMetrica LOGO, {r} test tracking, vs Metrica labels: " + " ".join(MET))
        for a, v in res.items():
            v = np.array(v[r])
            print(f"  {a:<4} {v.mean():.3f} | {' '.join(f'{x:+.3f}' for x in v - b)}  up {int((v > b).sum())}/3")
    json.dump(res, open(os.path.join(ex.OUT, "exp_weight.json"), "w"), indent=1)
    print("WEIGHT_DONE")
