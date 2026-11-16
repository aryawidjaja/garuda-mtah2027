"""On-par check against a published external score: CLEAN Metrica tracking (as delivered, no degradation, no cue),
leave-one-game-out, official spellF1 against our event-proxy labels. Folds as exp_seq.py's Metrica folds (train on
all DFL + the other two Metrica games, weight 5); decoder = shipped DFL smoother. Readouts: v3, GRU, blend.
Not a model-selection run: nothing is chosen from it.

    python garuda/scripts/exp_onpar.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import exp_seq as es  # noqa: E402
import experiment_metrica as em  # noqa: E402
from exp_w20 import DFL, MET, SC, decode_score, dfl_rows, eb, ex, ext_data, fit, met_rows, smoothers  # noqa: E402
from garuda import cue_transplant  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("CUE_LIB"):   # prebuilt library (npz of the live/dead patterns), so the raw SkillCorner files can stay home
        z = np.load(os.environ["CUE_LIB"])
        lib = {a: [z[k] for k in sorted((k for k in z.files if k.startswith(f"{int(a)}_")), key=lambda k: int(k[2:]))]
               for a in (True, False)}
    else:
        lib = cue_transplant.build_library(SC)
    Xd, yd = dfl_rows(DFL, lib)
    res = {"v3": {}, "gru": {}, "blend": {}}
    for m in MET:
        oth = [o for o in MET if o != m]
        mX, my = [], []
        for o in oth:
            xx, yy = met_rows(o, MET.index(o), lib); mX += xx; my += yy
        ws = [1] * len(Xd) + [5] * len(mX)
        clf, prior = fit(Xd + mX, yd + my, ws)
        gru = es.train(Xd + mX, yd + my, ws)
        del mX, my
        X0, X1, _, per, n25 = em.raw_features(m)
        X0, X1 = np.hstack([X0, eb.nan_block(len(X0))]), np.hstack([X1, eb.nan_block(len(X1))])
        pl = (clf.predict_proba(X0) + clf.predict_proba(X1)[:, [1, 0, 2]]) / 2
        pg = (es.predict(gru, X0, per) + es.predict(gru, X1, per)[:, [1, 0, 2]]) / 2
        gts = [s["label"].astype(np.int8) for s in ex.periods(ext_data.load_match(m))]
        h = smoothers(DFL, oth)["dfl"]
        res["v3"][m] = decode_score(pl, prior, h, per, n25, gts)
        res["gru"][m] = decode_score(pg, gru[3], h, per, n25, gts)
        res["blend"][m] = decode_score((pl + pg) / 2, prior, h, per, n25, gts)
        print(m, {a: round(r[m], 4) for a, r in res.items()}, flush=True)
    print("MEAN", {a: round(float(np.mean(list(r.values()))), 4) for a, r in res.items()})
    json.dump(res, open(os.path.join(ex.OUT, "exp_onpar.json"), "w"), indent=1)
    print("ONPAR_DONE")
