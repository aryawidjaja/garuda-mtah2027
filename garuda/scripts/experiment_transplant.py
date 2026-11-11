"""Experiment G: value of the TV-feed cue when tracking is independent of the broadcast (as on
BAR-ATH). DFL dev leave-one-match-out; the cue is transplanted from real SkillCorner camera
behaviour (library built from the 15 SkillCorner training games only). Paired with / without.

    python garuda/scripts/experiment_transplant.py
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
import experiment_sources as es  # noqa: E402
from garuda import broadcast, cue_transplant, ext_data  # noqa: E402


def cue_blocks(mid, seed, lib):
    rng = np.random.default_rng(seed + 777)
    return [broadcast.features(cue_transplant.transplant(sub["label"].astype(np.int8), lib, rng))[1]
            for sub in ex.periods(ext_data.load_match(mid))]


def with_cue(X, per, sw, blocks):
    B = np.full((len(X), blocks[0].shape[1]), np.nan, np.float32)
    for k, b in enumerate(blocks):
        for s in (False, True):
            m = (per == k) & (sw == s)
            if m.any():
                B[m] = b[:m.sum()]
    return np.hstack([X, B])


if __name__ == "__main__":
    dev = ext_data.list_matches("idsse")[:5]
    lib = cue_transplant.build_library(ext_data.list_matches("skillcorner")[:15])
    res = {}
    for m in dev:
        tr = [x for x in dev if x != m]
        h, _ = ex.hsmm_from_labels(tr, (1, 1, 1), 1.0)
        X0s, X1s, ys = [], [], []
        for s in tr:
            i = dev.index(s)
            for k in range(ex.K_DRAWS):
                _, X, y, per, _, sw = ex.match_features(s, 100 * i + k, True)
                X0s.append(X); ys.append(y)
                X1s.append(with_cue(X, per, sw, cue_blocks(s, 100 * i + k, lib)))
        c0, p0 = em.fit(X0s, ys)
        c1, p1 = em.fit(X1s, ys)
        _, Xt, _, per, n25, swt = ex.match_features(m, ex.TEST_SEED, False)
        _, Xa, _, pera, _, swa = ex.match_features(m, ex.TEST_SEED, True)
        blocks = cue_blocks(m, ex.TEST_SEED, lib)
        gts = ex.gt_parts(m)
        v0, _ = es.decode_parts(c0, p0, Xt, Xa[swa], per, n25, h)
        v1, _ = es.decode_parts(c1, p1, with_cue(Xt, per, swt, blocks),
                                with_cue(Xa, pera, swa, blocks)[swa], per, n25, h)
        res[m] = {"without": ex.score(gts, v0), "with_cue": ex.score(gts, v1)}
        print(m, {k: round(v["spell_f1_mean"], 4) for k, v in res[m].items()},
              "D-F1", {k: round(v["D_f1"], 3) for k, v in res[m].items()}, flush=True)
    json.dump(res, open(os.path.join(ex.OUT, "transplant_cue.json"), "w"), indent=1)
    for k in ("without", "with_cue"):
        print(f"{k:9s} spellF1 {np.mean([res[t][k]['spell_f1_mean'] for t in res]):.4f}  "
              f"D-F1 {np.mean([res[t][k]['D_f1'] for t in res]):.4f}")
    print("TRANSPLANT_DONE")
