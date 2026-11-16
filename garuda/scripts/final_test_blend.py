"""Gate for v4 = mean(v3 LightGBM, GRU) probabilities (exp_seq.py blend): the locked DFL pair, scored ONCE.

Train on the 5 dev DFL x 3 degraded draws + 3 Metrica games (clean + 3 degraded draws, weight 5), TV-feed cue
everywhere, as exp_w20.py / exp_seq.py; decoder = shipped DFL smoother from the 5 dev matches. Test: the 2 locked
DFL matches, one degraded draw (TEST_SEED) + cue. v3 (LightGBM alone) is scored in the same run.
Predeclared: ship the blend only if its pair mean is >= 0.665 AND >= v3's pair mean from this run.

    python garuda/scripts/final_test_blend.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import exp_seq as es  # noqa: E402
from exp_w20 import DEV, DFL, MET, SC, decode_score, dfl_rows, et, ex, fit, met_rows, smoothers  # noqa: E402
from garuda import cue_transplant  # noqa: E402


def test_rows(m, lib):
    """(X plain, X swapped, per, n25, 25 Hz label parts) for one locked DFL match: one degraded draw + cue."""
    tb = et.cue_blocks(m, ex.TEST_SEED, lib)
    _, X0, _, per, n25, sw0 = ex.match_features(m, ex.TEST_SEED, False)
    _, Xa, _, pera, _, swa = ex.match_features(m, ex.TEST_SEED, True)
    return et.with_cue(X0, per, sw0, tb), et.with_cue(Xa, pera, swa, tb)[swa], per, n25, ex.gt_parts(m)


if __name__ == "__main__":
    lib = cue_transplant.build_library(SC)
    test = DFL[5:]
    Xd, yd = dfl_rows(DEV, lib)
    mX, my = [], []
    for i, m in enumerate(MET):
        xx, yy = met_rows(m, i, lib); mX += xx; my += yy
    ws = [1] * len(Xd) + [5] * len(mX)
    clf, prior = fit(Xd + mX, yd + my, ws)
    print("lgbm fitted", flush=True)
    gru = es.train(Xd + mX, yd + my, ws)
    del Xd, mX
    h = smoothers(DEV, MET)["dfl"]
    res = {"v3": {}, "gru": {}, "blend": {}}
    for m in test:
        X0, X1, per, n25, gts = test_rows(m, lib)
        pl = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
        pg = (es.predict(gru, X0, per) + es.predict(gru, X1, per)[:, [1, 0, 2]]) / 2
        res["v3"][m] = decode_score(pl, prior, h, per, n25, gts)
        res["gru"][m] = decode_score(pg, gru[3], h, per, n25, gts)
        res["blend"][m] = decode_score((pl + pg) / 2, prior, h, per, n25, gts)
        print(m, {a: round(r[m], 4) for a, r in res.items()}, flush=True)
    mean = {a: float(np.mean(list(r.values()))) for a, r in res.items()}
    ok = mean["blend"] >= 0.665 and mean["blend"] >= mean["v3"]
    print("MEAN", {a: round(v, 4) for a, v in mean.items()}, f"gate (blend >= 0.665 and >= v3): {'PASS' if ok else 'FAIL'}")
    json.dump(res, open(os.path.join(ex.OUT, "final_test_blend.json"), "w"), indent=1)
    print("FINAL_BLEND_DONE")
