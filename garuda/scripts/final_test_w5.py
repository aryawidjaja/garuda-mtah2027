"""Gate 2 for the Metrica-weighted recipe (exp_weight.py W5): the locked DFL pair, scored ONCE.

Identical to final_test.py (v1 features, no TV cue, HSMM from the 5 dev DFL matches, Viterbi 1 s, same test
draw) except the training mix: 5 dev DFL x 3 degraded draws + the 3 Metrica games clean AND 3 degraded draws
each, Metrica rows weighted 5, no SkillCorner labels. Reference: the frozen system's 0.666 on this pair.
Predeclared: proceed if the mean is >= 0.656.

    python garuda/scripts/final_test_w5.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment_sources as es  # noqa: E402
from exp_weight import MET, both_views, em, ex, fit, met_degraded  # noqa: E402
from garuda import ext_data  # noqa: E402

if __name__ == "__main__":
    dfl = ext_data.list_matches("idsse"); dev, test = dfl[:5], dfl[5:]
    mX, my = [], []
    for i, m in enumerate(MET):
        xx, yy = both_views(*em.raw_features(m)[:3]); mX += xx; my += yy
        for k in range(ex.K_DRAWS):
            xx, yy = both_views(*met_degraded(m, 500 + 10 * i + k)[:3]); mX += xx; my += yy
    Xd, yd = [], []
    for i, s in enumerate(dev):
        for k in range(ex.K_DRAWS):
            _, X, y, _, _, _ = ex.match_features(s, 100 * i + k, True); Xd.append(X); yd.append(y)
    h, _ = ex.hsmm_from_labels(dev, (1, 1, 1), 1.0)
    clf, prior = fit(Xd + mX, yd + my, [1] * len(Xd) + [5] * len(mX))
    res = {}
    for m in test:
        _, X0, _, per, n25, _ = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, _, _, sw = ex.match_features(m, ex.TEST_SEED, True)
        vit, _ = es.decode_parts(clf, prior, X0, Xa[sw], per, n25, h)
        r = ex.score(ex.gt_parts(m), vit)
        res[m] = r
        print(f"W5 {m}: spellF1 {r['spell_f1_mean']:.4f} macroF1 {r['macro_f1']:.4f} D-F1 {r['D_f1']:.4f} "
              f"spells {r['n_pred_spells']}/{r['n_gt_spells']}", flush=True)
    mean = np.mean([res[m]["spell_f1_mean"] for m in test])
    print(f"W5 MEAN spellF1 {mean:.4f} (frozen 0.666; gate >= 0.656: {'PASS' if mean >= 0.656 else 'FAIL'})")
    json.dump(res, open(os.path.join(ex.OUT, "final_test_w5.json"), "w"), indent=1)
    print("FINAL_W5_DONE")
