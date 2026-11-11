"""External-data experiment: train on degraded DFL/IDSSE matches, evaluate on held-out ones with
the official scorer. No competition data is touched here.

Protocol
  dev  = 5 matches, leave-one-match-out (model and decoder choices are made here only)
  test = 2 matches, scored once at the end with the frozen choices
Each held-out match is degraded with a fixed seed that no training draw uses.

    python garuda/scripts/experiment.py --stage dev
    python garuda/scripts/experiment.py --stage test
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "official-repo", "evaluation"))
from garuda import decode, degrade, ext_data, features, relabel, tags  # noqa: E402
import possession_metrics as pm  # noqa: E402

CACHE = os.path.join(ROOT, "external", "cache")
OUT = os.path.join(ROOT, "garuda", "results")
K_DRAWS = 3
TEST_SEED = 10_000
TARGET = os.environ.get("GARUDA_TARGET", "relabel")   # "relabel" (controlled possession) or "raw"


def target_label(sub):
    lab = sub["label"].astype(np.int8)
    if TARGET == "raw":
        return lab
    return relabel.relabel(lab, sub["ball_xy"], sub["players_xy"], sub["players_team"],
                           sub.get("ball_z"))


def periods(d):
    out = []
    for p in np.unique(d["period"]):
        idx = np.flatnonzero(d["period"] == p)
        sub = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(d["period"]) else v)
               for k, v in d.items()}
        sub["pitch"] = (d["meta"]["pitch_length"], d["meta"]["pitch_width"])
        out.append(sub)
    return out


def match_features(mid, seed, swap_aug):
    """Features/labels at 5 Hz for every period of one match, one degradation draw."""
    v = "" if features.VERSION == 1 else f"_v{features.VERSION}"
    path = os.path.join(CACHE, f"{mid}_{TARGET}_s{seed}_{int(swap_aug)}{v}.npz")
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=True)
        return list(z["names"]), z["X"], z["y"], z["per"], z["n25"], z["sw"]
    d = ext_data.load_match(mid)
    rng = np.random.default_rng(seed)
    Xs, ys, pers, n25, sws = [], [], [], [], []
    for k, sub in enumerate(periods(d)):
        p = degrade.sample_params(rng)
        dd = degrade.degrade(sub, sub["label"], p, rng)
        tl = target_label(sub)
        for swap in ((False, True) if swap_aug else (False,)):
            names, X = features.build(dd, swap=swap)
            y = features.pool_labels(tl)
            if swap:
                y = np.where(y == 2, 2, 1 - y)
            Xs.append(X); ys.append(y); pers.append(np.full(len(y), k)); sws.append(np.full(len(y), swap))
        n25.append(len(sub["label"]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, names=np.array(names), X=np.concatenate(Xs), y=np.concatenate(ys),
                        per=np.concatenate(pers), n25=np.array(n25), sw=np.concatenate(sws))
    return (names, np.concatenate(Xs), np.concatenate(ys), np.concatenate(pers), np.array(n25),
            np.concatenate(sws))


def train(mids, seed0=0):
    import lightgbm as lgb
    Xs, ys = [], []
    for i, m in enumerate(mids):
        for k in range(K_DRAWS):
            _, X, y, _, _, _ = match_features(m, seed0 + 100 * i + k, True)
            Xs.append(X); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             min_child_samples=200, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.6, reg_lambda=1.0, n_jobs=4, verbose=-1,
                             random_state=0, deterministic=True)
    clf.fit(X, y)
    return clf, np.bincount(y, minlength=3) / len(y)


def predict_tta(clf, mid, seed):
    """Average the plain and the team-swapped view (swap probs mapped back)."""
    names, X, y, per, n25, _ = match_features(mid, seed, False)
    _, Xa, _, _, _, sw = match_features(mid, seed, True)
    Xs = Xa[sw]                            # swapped view, same degradation draw, same order
    p1 = clf.predict_proba(X)
    p2 = clf.predict_proba(Xs)[:, [1, 0, 2]]
    return (p1 + p2) / 2, y, per, n25


def hsmm_from_labels(mids, min_s, alpha):
    """Durations and switch matrix from training labels (5 Hz)."""
    durs = {0: [], 1: [], 2: []}
    trans = np.ones((3, 3))
    for m in mids:
        d = ext_data.load_match(m)
        for sub in periods(d):
            y = features.pool_labels(target_label(sub))
            r = ext_data.runs(y)
            for (s, e, v), nxt in zip(r, r[1:] + [None]):
                durs[v].append(e - s)
                if nxt:
                    trans[v, nxt[2]] += 1
    md = [max(1, int(round(min_s[L] * 5))) for L in range(3)]
    stay = [1 - 1 / max(np.mean(durs[L]) - md[L] + 1, 2) for L in range(3)]
    m = decode.HSMM(md, stay, trans)
    m.switch_matrix, m.stay_probs = trans.tolist(), [float(x) for x in stay]
    return m, alpha


def to25(lab5, n):
    return np.repeat(lab5, features.STEP)[:n] if len(lab5) * features.STEP >= n else \
        np.concatenate([np.repeat(lab5, features.STEP), np.full(n - len(lab5) * features.STEP, lab5[-1])])


def score(gt_parts, pr_parts):
    sep = np.array([pm.X])
    gt = np.concatenate(sum([[g.astype(np.int64), sep] for g in gt_parts], [])[:-1])
    pr = np.concatenate(sum([[p.astype(np.int64), sep] for p in pr_parts], [])[:-1])
    r = pm.evaluate(gt, pr)
    return {k: r[k] for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells",
                              "n_pred_spells", "accuracy")} | {
        "D_f1": r["per_class"]["D"]["f1"], "D_recall": r["per_class"]["D"]["recall"]}


def baseline_parts(mid, seed):
    """Official baseline logic (nearest player, +/-50 majority vote) on the same degraded draw."""
    d = ext_data.load_match(mid)
    rng = np.random.default_rng(seed)
    parts = []
    for sub in periods(d):
        dd = degrade.degrade(sub, sub["label"], degrade.sample_params(rng), rng)
        b, pl, tm = dd["ball_xy"], dd["players_xy"], dd["players_team"]
        dist = np.hypot(pl[..., 0] - b[:, None, 0], pl[..., 1] - b[:, None, 1])
        dist = np.where(np.isnan(dist), np.inf, dist)
        code = np.where(np.isnan(b[:, 0]) | ~np.isfinite(dist.min(1)), -1, tm[dist.argmin(1)])
        h = np.concatenate(([0], np.cumsum(code == 0)))
        a = np.concatenate(([0], np.cumsum(code == 1)))
        i = np.arange(len(code))
        lo, hi = np.maximum(0, i - 50), np.minimum(len(code), i + 51)
        hh, aa = h[hi] - h[lo], a[hi] - a[lo]
        parts.append(np.where(hh + aa == 0, 2, np.where(hh >= aa, 0, 1)))
    return parts


def gt_parts(mid):
    return [target_label(sub) for sub in periods(ext_data.load_match(mid))]


def sim_tag_bonus(mid, n5s, beta, gamma, seed):
    """Metrica-like set-piece windows simulated from the held-out match's own restarts."""
    rng = np.random.default_rng(seed)
    out = []
    for sub, n5 in zip(periods(ext_data.load_match(mid)), n5s):
        w = tags.simulate(sub["label"].astype(np.int8), rng)
        out.append(tags.bonus_from_windows(w, n5, beta, gamma, None))
    return out


def run_fold(train_mids, test_mid, grid):
    t0 = time.time()
    clf, prior = train(train_mids)
    proba, y5, per, n25 = predict_tta(clf, test_mid, TEST_SEED)
    gts = gt_parts(test_mid)
    res = {"baseline_smooth50": score(gts, baseline_parts(test_mid, TEST_SEED))}
    res["argmax"] = score(gts, [to25(proba[per == k].argmax(1), n25[k]) for k in range(len(n25))])
    n5s = [int((per == k).sum()) for k in range(len(n25))]
    for min_s, alpha, beta, gamma in grid:
        m, a = hsmm_from_labels(train_mids, min_s, alpha)
        tb = sim_tag_bonus(test_mid, n5s, beta, gamma, TEST_SEED + 1) if (beta or gamma) else None
        vit_parts, mbr_parts = [], []
        for k in range(len(n25)):
            e = decode.emissions(proba[per == k], prior, a)
            if tb is not None:
                e = e + tb[k]
            vit = m.viterbi(e)
            smp = m.sample_paths(e, 24, seed=k)
            cands = [vit] + list(smp)
            best, _ = decode.mbr_select(cands, smp)
            vit_parts.append(to25(vit, n25[k]))
            mbr_parts.append(to25(cands[best], n25[k]))
        tag = f"ms{'-'.join(str(x) for x in min_s)}_a{alpha}_t{beta}-{gamma}"
        res[f"viterbi_{tag}"] = score(gts, vit_parts)
        res[f"mbr_{tag}"] = score(gts, mbr_parts)
    res["_seconds"] = time.time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dev", "test"), default="dev")
    args = ap.parse_args()
    mids = ext_data.list_matches("idsse")
    assert len(mids) == 7, mids
    dev, test = mids[:5], mids[5:]
    grid = [((0.2, 0.2, 0.2), 1.0, 0, 0), ((1, 1, 1), 0.5, 0, 0), ((1, 1, 1), 1.0, 0, 0),
            ((2, 2, 2), 0.5, 0, 0), ((1, 1, 2), 0.7, 0, 0), ((1, 1, 1), 0.5, 1.5, 1.0),
            ((1, 1, 1), 0.5, 3.0, 2.0)]
    os.makedirs(OUT, exist_ok=True)
    if args.stage == "dev":
        allres = {}
        for m in dev:
            allres[m] = run_fold([x for x in dev if x != m], m, grid)
            print(m, json.dumps({k: round(v["spell_f1_mean"], 4) for k, v in allres[m].items()
                                 if isinstance(v, dict)}), flush=True)
        json.dump(allres, open(os.path.join(OUT, "dev_lomo.json"), "w"), indent=1)
    else:
        cfg = json.load(open(os.path.join(OUT, "frozen.json")))
        allres = {m: run_fold(dev, m, [(tuple(cfg["min_s"]), cfg["alpha"], cfg["tag_beta"],
                                         cfg["tag_gamma"])]) for m in test}
        json.dump(allres, open(os.path.join(OUT, "test.json"), "w"), indent=1)
    keys = [k for k in next(iter(allres.values())) if not k.startswith("_")]
    print("\nmean over held-out matches:")
    for k in keys:
        v = np.array([[allres[m][k]["spell_f1_mean"], allres[m][k]["macro_f1"], allres[m][k]["D_f1"]]
                      for m in allres])
        print(f"{k:28s} spellF1 {v[:,0].mean():.4f}  macroF1 {v[:,1].mean():.4f}  D-F1 {v[:,2].mean():.4f}")
