"""Experiment E: do extra real-tracking sources help?
  C0  DFL (degraded) only (current system)
  C1  DFL (degraded) + 15 SkillCorner + 3 Metrica (as delivered: already realistic gaps)
Scored paired on the same held-out DFL dev matches, and on 5 SkillCorner games neither saw.

    python garuda/scripts/experiment_sources.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import experiment as ex  # noqa: E402
import experiment_metrica as em  # noqa: E402
from garuda import decode, ext_data, features, sources  # noqa: E402


def sc_features(mid):
    v = "" if features.VERSION == 1 else f"_v{features.VERSION}"
    path = os.path.join(ex.CACHE, f"{mid}_native{v}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return z["X0"], z["X1"], z["y"], z["per"], list(z["n25"])
    X0, X1, ys, pers, n25 = [], [], [], [], []
    for k, sub in enumerate(sources.skillcorner_periods(mid)):
        _, a = features.build(sub)
        _, b = features.build(sub, swap=True)
        y = features.pool_labels(sub["label"])
        X0.append(a); X1.append(b); ys.append(y); pers.append(np.full(len(y), k)); n25.append(len(sub["label"]))
    out = (np.concatenate(X0), np.concatenate(X1), np.concatenate(ys), np.concatenate(pers), n25)
    np.savez_compressed(path, X0=out[0], X1=out[1], y=out[2], per=out[3], n25=np.array(n25))
    return out


def stack(rows):
    """[(X0, X1, y)] -> training rows with the swapped view and flipped labels."""
    Xs, ys = [], []
    for X0, X1, y in rows:
        Xs += [X0, X1]
        ys += [y, np.where(y == 2, 2, 1 - y)]
    return Xs, ys


def decode_parts(clf, prior, X0, X1, per, n25, h):
    p = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
    vit, mbr = [], []
    for k in range(len(n25)):
        e = decode.emissions(p[per == k], prior, 1.0)
        v = h.viterbi(e)
        s = h.sample_paths(e, 24, seed=k)
        c = [v] + list(s)
        vit.append(ex.to25(v, n25[k]))
        mbr.append(ex.to25(c[decode.mbr_select(c, s)[0]], n25[k]))
    return vit, mbr


def sc_gt(mid):
    return [s["label"] for s in sources.skillcorner_periods(mid)]


if __name__ == "__main__":
    dfl = ext_data.list_matches("idsse")
    dev = dfl[:5]
    sc = ext_data.list_matches("skillcorner")
    sc_train, sc_test = sc[:15], sc[15:]
    extra = [sc_features(m)[:3] for m in sc_train]
    extra += [em.raw_features(m)[:3] for m in ext_data.list_matches("metrica")]
    ex_X, ex_y = stack(extra)
    res = {}
    for m in dev:
        tr = [x for x in dev if x != m]
        h, _ = ex.hsmm_from_labels(tr, (1, 1, 1), 1.0)
        Xd, yd = [], []
        for i, s in enumerate(tr):
            for k in range(ex.K_DRAWS):
                _, X, y, _, _, _ = ex.match_features(s, 100 * dev.index(s) + k, True)
                Xd.append(X); yd.append(y)
        _, X0, y0, per, n25, _ = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, _, _, sw = ex.match_features(m, ex.TEST_SEED, True)
        gts = ex.gt_parts(m)
        res[m] = {}
        for name, (Xs, ys) in (("C0", (Xd, yd)), ("C1", (Xd + ex_X, yd + ex_y))):
            clf, prior = em.fit(Xs, ys)
            vit, mbr = decode_parts(clf, prior, X0, Xa[sw], per, n25, h)
            res[m][f"{name}_vit"] = ex.score(gts, vit)
            res[m][f"{name}_mbr"] = ex.score(gts, mbr)
            if m == dev[0]:          # SkillCorner hold-out: once, with this fold's models
                for t in sc_test:
                    a, b, _, pp, nn = sc_features(t)
                    v, mb = decode_parts(clf, prior, a, b, pp, nn, h)
                    res.setdefault("sc_test", {}).setdefault(name, []).append(
                        (ex.score(sc_gt(t), v)["spell_f1_mean"], ex.score(sc_gt(t), mb)["spell_f1_mean"]))
        print(m, {k: round(v["spell_f1_mean"], 4) for k, v in res[m].items()}, flush=True)
    json.dump(res, open(os.path.join(ex.OUT, f"sources_{ex.TARGET}.json"), "w"), indent=1)
    print("\nDFL dev mean (paired):")
    for k in ("C0_vit", "C0_mbr", "C1_vit", "C1_mbr"):
        print(f"{k}  spellF1 {np.mean([res[m][k]['spell_f1_mean'] for m in dev]):.4f}")
    for name in ("C0", "C1"):
        a = np.array(res["sc_test"][name])
        print(f"SkillCorner hold-out {name}: viterbi {a[:,0].mean():.4f}  mbr {a[:,1].mean():.4f}")
