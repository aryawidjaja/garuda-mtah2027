"""W5 with the TV-feed cue as shipped: does the cue path survive dropping SkillCorner label rows?

DFL dev LOMO as exp_weight_dev.py, plus the 8 cue columns: DFL draws and degraded Metrica draws get the
SkillCorner camera-behaviour transplant (cue_transplant library from all 20 SkillCorner games, camera only, no
labels), clean Metrica rows get NaN (as train_final.py), the held-out test draw gets a transplant too.
Readout against W5 without the cue (0.627 in exp_weight_dev_s.py). Predeclared: ship W5 with the cue if
its mean is >= 0.617 (no cue minus 0.01); otherwise ship W5 without the cue.

    python garuda/scripts/exp_weight_cue.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment_broadcast as eb, experiment_transplant as et  # noqa: E402
from exp_weight import MET, SC, both_views, em, ex, fit, met_degraded  # noqa: E402
from garuda import cue_transplant, decode, ext_data  # noqa: E402


def met_cue(mid, seed, lib, X0, X1, per):
    """Cue columns for a degraded Metrica draw (same rows for the plain and swapped views)."""
    blocks = et.cue_blocks(mid, seed, lib)
    B = np.full((len(X0), blocks[0].shape[1]), np.nan, np.float32)
    for k, b in enumerate(blocks):
        m = per == k
        B[m] = b[:m.sum()]
    return np.hstack([X0, B]), np.hstack([X1, B])


DEV = ext_data.list_matches("idsse")[:5]
if __name__ == "__main__":
    lib = cue_transplant.build_library(SC)
    mX, my = [], []
    for i, m in enumerate(MET):
        X0, X1, y, _, _ = em.raw_features(m)
        xx, yy = both_views(np.hstack([X0, eb.nan_block(len(X0))]), np.hstack([X1, eb.nan_block(len(X1))]), y)
        mX += xx; my += yy
        for k in range(ex.K_DRAWS):
            seed = 500 + 10 * i + k
            X0, X1, y, per, _ = met_degraded(m, seed)
            xx, yy = both_views(*met_cue(m, seed, lib, X0, X1, per), y); mX += xx; my += yy
    res = []
    for m in DEV:
        tr = [x for x in DEV if x != m]
        Xd, yd = [], []
        for s in tr:
            for k in range(ex.K_DRAWS):
                seed = 100 * DEV.index(s) + k
                _, X, y, per, _, sw = ex.match_features(s, seed, True)
                Xd.append(et.with_cue(X, per, sw, et.cue_blocks(s, seed, lib))); yd.append(y)
        clf, prior = fit(Xd + mX, yd + my, [1] * len(Xd) + [5] * len(mX))
        h, _ = ex.hsmm_from_labels(tr, (1, 1, 1), 1.0)
        tb = et.cue_blocks(m, ex.TEST_SEED, lib)
        _, X0, _, per, n25, sw0 = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, pera, _, swa = ex.match_features(m, ex.TEST_SEED, True)
        X0, Xa = et.with_cue(X0, per, sw0, tb), et.with_cue(Xa, pera, swa, tb)
        p = (clf.predict_proba(X0) + clf.predict_proba(Xa[swa])[:, [1, 0, 2]]) / 2
        parts = [ex.to25(h.viterbi(decode.emissions(p[per == k], prior, 1.0)), n) for k, n in enumerate(n25)]
        sc = ex.score(ex.gt_parts(m), parts)
        res.append([sc["spell_f1_mean"], sc["macro_f1"], sc["D_f1"]])
        print("done", m, [round(x, 3) for x in res[-1]], flush=True)
    v = np.array(res)
    print(f"\nW5 + cue: spellF1 {v[:, 0].mean():.3f} macroF1 {v[:, 1].mean():.3f} D-F1 {v[:, 2].mean():.3f} "
          f"(W5 no cue 0.627 / 0.884 / 0.936) -> {'SHIP WITH CUE' if v[:, 0].mean() >= 0.617 else 'SHIP WITHOUT CUE'}")
    json.dump(res, open(os.path.join(ex.OUT, "exp_weight_cue.json"), "w"), indent=1)
    print("WEIGHT_CUE_DONE")
