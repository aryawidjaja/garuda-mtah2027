"""Score the frozen configuration ONCE on the 2 DFL matches no choice has touched.

    python garuda/scripts/final_test.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment as ex, experiment_metrica as em, experiment_sources as es  # noqa: E402
from garuda import ext_data  # noqa: E402

if __name__ == "__main__":
    dfl = ext_data.list_matches("idsse"); dev, test = dfl[:5], dfl[5:]
    sc = ext_data.list_matches("skillcorner")
    extra = [es.sc_features(m)[:3] for m in sc] + [em.raw_features(m)[:3] for m in ext_data.list_matches("metrica")]
    ex_X, ex_y = es.stack(extra)
    Xd, yd = [], []
    for i, s in enumerate(dev):
        for k in range(ex.K_DRAWS):
            _, X, y, _, _, _ = ex.match_features(s, 100 * i + k, True); Xd.append(X); yd.append(y)
    h, _ = ex.hsmm_from_labels(dev, (1, 1, 1), 1.0)
    res = {}
    for name, (Xs, ys) in (("frozen_C1", (Xd + ex_X, yd + ex_y)), ("reference_C0", (Xd, yd))):
        clf, prior = em.fit(Xs, ys)
        for m in test:
            _, X0, _, per, n25, _ = ex.match_features(m, ex.TEST_SEED, False)
            _, Xa, _, _, _, sw = ex.match_features(m, ex.TEST_SEED, True)
            vit, _ = es.decode_parts(clf, prior, X0, Xa[sw], per, n25, h)
            res.setdefault(name, {})[m] = ex.score(ex.gt_parts(m), vit)
            res[name][m + "_baseline"] = ex.score(ex.gt_parts(m), ex.baseline_parts(m, ex.TEST_SEED))
    json.dump(res, open(os.path.join(ex.OUT, "final_test.json"), "w"), indent=1)
    for name in res:
        for m in test:
            r, b = res[name][m], res[name][m + "_baseline"]
            print(f"{name:13s} {m}: spellF1 {r['spell_f1_mean']:.4f} macroF1 {r['macro_f1']:.4f} "
                  f"D-F1 {r['D_f1']:.4f} spells {r['n_pred_spells']}/{r['n_gt_spells']} | baseline {b['spell_f1_mean']:.4f}")
        print(f"{name:13s} MEAN spellF1 {np.mean([res[name][m]['spell_f1_mean'] for m in test]):.4f}")
    print("FINAL_TEST_DONE")
