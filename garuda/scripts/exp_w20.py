"""Experiment W20 + smoothing fit, on the v3 recipe (TV-feed cue everywhere), arms W5 (= v3) and W20 trained by
the identical code, so the comparison is paired and needs no logged baseline:

  DFL dev LOMO (5 folds), test = one degraded draw vs DFL relabel labels      (W5 must reproduce 0.643)
  Metrica LOGO (3 folds), test = one degraded draw + cue vs Metrica labels
Per-frame probabilities of every fold and arm go to results/exp_w20_proba.npz, then each is decoded with three
semi-Markov smoothers whose durations and switch counts come from the fold's TRAINING games only:
  dfl  DFL relabel labels (as shipped)   met  Metrica labels   mix  DFL + Metrica runs counted 5x

Predeclared gate for any change against W5 + dfl (v3), both readouts:
  Metrica mean >= +0.005 and up on >= 2/3 games, and DFL mean >= -0.005.
W20 passing also needs the locked pair (final_test_w5.py with weight 20) >= 0.665 (W5 0.675 - 0.01).

    python garuda/scripts/exp_w20.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment_broadcast as eb, experiment_transplant as et  # noqa: E402
from exp_weight import MET, SC, TEST_DRAW, both_views, em, ex, fit, met_degraded  # noqa: E402
from exp_weight_cue import met_cue  # noqa: E402
from garuda import cue_transplant, decode, ext_data, features  # noqa: E402

DFL = ext_data.list_matches("idsse")
DEV = DFL[:5]
ARMS = {"W5": 5, "W20": 20}
PROBA = os.path.join(ex.OUT, "exp_w20_proba.npz")


def hsmm(runs_w):
    """Semi-Markov smoother from [(5 Hz label array, weight)], as ex.hsmm_from_labels with min 1 s."""
    durs, trans = {0: [], 1: [], 2: []}, np.ones((3, 3))
    for y, w in runs_w:
        r = ext_data.runs(y)
        for (s, e, v), nxt in zip(r, r[1:] + [None]):
            durs[v] += [e - s] * w
            if nxt:
                trans[v, nxt[2]] += w
    stay = [1 - 1 / max(np.mean(durs[L]) - 5 + 1, 2) for L in range(3)]
    return decode.HSMM([5, 5, 5], stay, trans)


def labels5(mid, relab):
    return [features.pool_labels(ex.target_label(s) if relab else s["label"].astype(np.int8))
            for s in ex.periods(ext_data.load_match(mid))]


def smoothers(dfl_tr, met_tr):
    d = [(y, 1) for m in dfl_tr for y in labels5(m, True)]
    me = [(y, 1) for m in met_tr for y in labels5(m, False)]
    return {"dfl": hsmm(d), "met": hsmm(me), "mix": hsmm(d + [(y, 5) for y, _ in me])}


def decode_score(p, prior, h, per, n25, gts):
    parts = [ex.to25(h.viterbi(decode.emissions(p[per == k], prior, 1.0)), n) for k, n in enumerate(n25)]
    return ex.score(gts, parts)["spell_f1_mean"]


def met_rows(m, i, lib):
    X0, X1, y, _, _ = em.raw_features(m)
    Xs, ys = both_views(np.hstack([X0, eb.nan_block(len(X0))]), np.hstack([X1, eb.nan_block(len(X1))]), y)
    for k in range(ex.K_DRAWS):
        seed = 500 + 10 * i + k
        X0, X1, y, per, _ = met_degraded(m, seed)
        xx, yy = both_views(*met_cue(m, seed, lib, X0, X1, per), y); Xs += xx; ys += yy
    return Xs, ys


def dfl_rows(mids, lib):
    Xd, yd = [], []
    for s in mids:
        for k in range(ex.K_DRAWS):
            seed = 100 * DFL.index(s) + k
            _, X, y, per, _, sw = ex.match_features(s, seed, True)
            Xd.append(et.with_cue(X, per, sw, et.cue_blocks(s, seed, lib))); yd.append(y)
    return Xd, yd


def run_fold(key, Xd, yd, mX, my, X0, X1, dump):
    for a, w in ARMS.items():
        if f"{key}|{a}|p" in dump:
            continue
        clf, prior = fit(Xd + mX, yd + my, [1] * len(Xd) + [w] * len(mX))
        dump[f"{key}|{a}|p"] = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
        dump[f"{key}|{a}|prior"] = prior
        np.savez_compressed(PROBA, **dump)
        print("fit", key, a, flush=True)


if __name__ == "__main__":
    lib = cue_transplant.build_library(SC)
    dump = dict(np.load(PROBA)) if os.path.isfile(PROBA) else {}
    folds = []                                       # (key, per, n25, gts, dfl_tr, met_tr)
    mX, my = [], []
    for i, m in enumerate(MET):
        xx, yy = met_rows(m, i, lib); mX += xx; my += yy
    for m in DEV:                                    # 1. DFL dev LOMO
        tr = [x for x in DEV if x != m]
        tb = et.cue_blocks(m, ex.TEST_SEED, lib)
        _, X0, _, per, n25, sw0 = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, pera, _, swa = ex.match_features(m, ex.TEST_SEED, True)
        X0, Xa = et.with_cue(X0, per, sw0, tb), et.with_cue(Xa, pera, swa, tb)
        if any(f"dfl:{m}|{a}|p" not in dump for a in ARMS):
            run_fold(f"dfl:{m}", *dfl_rows(tr, lib), mX, my, X0, Xa[swa], dump)
        folds.append((f"dfl:{m}", per, n25, ex.gt_parts(m), tr, MET))
    del mX, my
    Xd, yd = dfl_rows(DFL, lib)                      # 2. Metrica LOGO
    for g, m in enumerate(MET):
        oth = [o for o in MET if o != m]
        X0, X1, _, per, n25 = met_degraded(m, TEST_DRAW + g)
        X0, X1 = met_cue(m, TEST_DRAW + g, lib, X0, X1, per)
        if any(f"met:{m}|{a}|p" not in dump for a in ARMS):
            mX, my = [], []
            for o in oth:
                xx, yy = met_rows(o, MET.index(o), lib); mX += xx; my += yy
            run_fold(f"met:{m}", Xd, yd, mX, my, X0, X1, dump)
        folds.append((f"met:{m}", per, n25, [s["label"].astype(np.int8) for s in ex.periods(ext_data.load_match(m))],
                      DFL, oth))

    res = {}
    for key, per, n25, gts, dtr, mtr in folds:
        hs = smoothers(dtr, mtr)
        for a in ARMS:
            for s, h in hs.items():
                res.setdefault(f"{a}+{s}", {})[key] = decode_score(dump[f"{key}|{a}|p"], dump[f"{key}|{a}|prior"],
                                                                   h, per, n25, gts)
        print(key, {k: round(v[key], 3) for k, v in res.items()}, flush=True)
    base = res["W5+dfl"]
    print("\narm        DFL dev  delta per match            | Metrica  delta per game       | gate")
    for arm, r in res.items():
        d = np.array([r[k] - base[k] for k in base if k.startswith("dfl:")])
        mm = np.array([r[k] - base[k] for k in base if k.startswith("met:")])
        dm = np.mean([r[k] for k in r if k.startswith("dfl:")]); mt = np.mean([r[k] for k in r if k.startswith("met:")])
        ok = mm.mean() >= 0.005 and (mm > 0).sum() >= 2 and d.mean() >= -0.005
        print(f"{arm:10s} {dm:.3f}  {' '.join(f'{x:+.3f}' for x in d)} | {mt:.3f}  {' '.join(f'{x:+.3f}' for x in mm)} | "
              f"{'-' if arm == 'W5+dfl' else ('PASS' if ok else 'FAIL')}")
    json.dump(res, open(os.path.join(ex.OUT, "exp_w20.json"), "w"), indent=1)
    print("W20_DONE")
