"""Experiment SEQ: a small bidirectional GRU over the v3 feature rows (230 columns, 5 Hz), same folds, rows,
Metrica weight 5 and test draws as exp_w20.py. It reads the whole timeline instead of one row and its fixed
windows. Readouts, each decoded with the shipped DFL smoother:
  seq     GRU probabilities alone
  blend   mean of GRU and v3 (LightGBM W5, from exp_w20_proba.npz)
Gate as exp_w20.py against W5 + dfl: Metrica mean >= +0.005 and up 2/3, DFL mean >= -0.005.

    python garuda/scripts/exp_seq.py        (after exp_w20.py)
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from exp_w20 import (DEV, DFL, MET, PROBA, SC, TEST_DRAW, decode_score, dfl_rows, et, ex, ext_data, met_cue,  # noqa: E402
                     met_degraded, met_rows, smoothers)
from garuda import cue_transplant, seq  # noqa: E402

torch.set_num_threads(2)                                  # heat: half the cores of the LightGBM runs
W_MET = 5
Net, train, predict = seq.Net, seq.train, seq.predict


if __name__ == "__main__":
    lib = cue_transplant.build_library(SC)
    lg = np.load(PROBA)
    res = {"W5+dfl": {}, "seq": {}, "blend": {}}

    def readout(key, model, X0, X1, per, n25, gts, dtr, mtr):
        p = (predict(model, X0, per) + predict(model, X1, per)[:, [1, 0, 2]]) / 2
        h = smoothers(dtr, mtr)["dfl"]
        pl, prl = lg[f"{key}|W5|p"], lg[f"{key}|W5|prior"]
        res["W5+dfl"][key] = decode_score(pl, prl, h, per, n25, gts)
        res["seq"][key] = decode_score(p, model[3], h, per, n25, gts)
        res["blend"][key] = decode_score((p + pl) / 2, prl, h, per, n25, gts)
        print(key, {a: round(r[key], 3) for a, r in res.items()}, flush=True)

    mX, my = [], []
    for i, m in enumerate(MET):
        xx, yy = met_rows(m, i, lib); mX += xx; my += yy
    for m in DEV:
        tr = [x for x in DEV if x != m]
        Xd, yd = dfl_rows(tr, lib)
        model = train(Xd + mX, yd + my, [1] * len(Xd) + [W_MET] * len(mX))
        tb = et.cue_blocks(m, ex.TEST_SEED, lib)
        _, X0, _, per, n25, sw0 = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, pera, _, swa = ex.match_features(m, ex.TEST_SEED, True)
        X0, Xa = et.with_cue(X0, per, sw0, tb), et.with_cue(Xa, pera, swa, tb)
        readout(f"dfl:{m}", model, X0, Xa[swa], per, n25, ex.gt_parts(m), tr, MET)
    del mX, my
    Xd, yd = dfl_rows(DFL, lib)
    for g, m in enumerate(MET):
        oth = [o for o in MET if o != m]
        mX, my = [], []
        for o in oth:
            xx, yy = met_rows(o, MET.index(o), lib); mX += xx; my += yy
        model = train(Xd + mX, yd + my, [1] * len(Xd) + [W_MET] * len(mX))
        X0, X1, _, per, n25 = met_degraded(m, TEST_DRAW + g)
        X0, X1 = met_cue(m, TEST_DRAW + g, lib, X0, X1, per)
        readout(f"met:{m}", model, X0, X1, per, n25,
                [s["label"].astype(np.int8) for s in ex.periods(ext_data.load_match(m))], DFL, oth)

    base = res["W5+dfl"]
    for arm, r in res.items():
        d = np.array([r[k] - base[k] for k in base if k.startswith("dfl:")])
        mm = np.array([r[k] - base[k] for k in base if k.startswith("met:")])
        ok = mm.mean() >= 0.005 and (mm > 0).sum() >= 2 and d.mean() >= -0.005
        print(f"{arm:7s} DFL {np.mean([r[k] for k in r if k.startswith('dfl:')]):.3f} {' '.join(f'{x:+.3f}' for x in d)} | "
              f"Metrica {np.mean([r[k] for k in r if k.startswith('met:')]):.3f} {' '.join(f'{x:+.3f}' for x in mm)} | "
              f"{'-' if arm == 'W5+dfl' else ('PASS' if ok else 'FAIL')}")
    json.dump(res, open(os.path.join(ex.OUT, "exp_seq.json"), "w"), indent=1)
    print("SEQ_DONE")
