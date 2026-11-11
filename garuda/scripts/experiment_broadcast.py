"""Experiment F: the TV-feed "not shown live" cue.
Training: frozen recipe (5 DFL dev degraded x3 + 15 SkillCorner + 3 Metrica). The cue exists only
where a broadcast camera exists (SkillCorner); elsewhere its columns are NaN, which LightGBM handles.
Test: 5 held-out SkillCorner games, paired with/without the cue.

    python garuda/scripts/experiment_broadcast.py
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
from garuda import broadcast, ext_data, sources  # noqa: E402

NBC = 8


def sc_bc(mid):
    """Cue columns aligned with experiment_sources.sc_features rows (same periods, same 25 Hz grid)."""
    d = ext_data.load_match(mid)
    live = broadcast.skillcorner_live(mid)
    out = []
    for p in np.unique(d["period"]):
        idx = np.flatnonzero(d["period"] == p)
        n25 = int((len(idx) - 1) * 2.5)
        if n25 < 100:
            continue
        nearest = np.minimum(np.round(np.arange(n25) / 2.5).astype(int), len(idx) - 1)
        out.append(broadcast.features(live[idx][nearest])[1])
    return np.concatenate(out)


def nan_block(n):
    return np.full((n, NBC), np.nan, np.float32)


if __name__ == "__main__":
    dfl = ext_data.list_matches("idsse")[:5]
    sc = ext_data.list_matches("skillcorner")
    sc_train, sc_test = sc[:15], sc[15:]
    h, _ = ex.hsmm_from_labels(dfl, (1, 1, 1), 1.0)
    base_X, base_y, cue_X = [], [], []
    for i, s in enumerate(dfl):
        for k in range(ex.K_DRAWS):
            _, X, y, _, _, _ = ex.match_features(s, 100 * i + k, True)
            base_X.append(X); base_y.append(y); cue_X.append(np.hstack([X, nan_block(len(X))]))
    for m in sc_train:
        X0, X1, y, _, _ = es.sc_features(m)
        b = sc_bc(m)
        assert len(b) == len(X0), (m, len(b), len(X0))
        base_X += [X0, X1]; base_y += [y, np.where(y == 2, 2, 1 - y)]
        cue_X += [np.hstack([X0, b]), np.hstack([X1, b])]
    for m in ext_data.list_matches("metrica"):
        X0, X1, y, _, _ = em.raw_features(m)
        base_X += [X0, X1]; base_y += [y, np.where(y == 2, 2, 1 - y)]
        cue_X += [np.hstack([X0, nan_block(len(X0))]), np.hstack([X1, nan_block(len(X1))])]
    clf0, prior0 = em.fit(base_X, base_y)
    clf1, prior1 = em.fit(cue_X, base_y)
    res = {}
    for t in sc_test:
        X0, X1, _, per, n25 = es.sc_features(t)
        b = sc_bc(t)
        gt = es.sc_gt(t)
        v0, _ = es.decode_parts(clf0, prior0, X0, X1, per, n25, h)
        v1, _ = es.decode_parts(clf1, prior1, np.hstack([X0, b]), np.hstack([X1, b]), per, n25, h)
        res[t] = {"without": ex.score(gt, v0), "with_cue": ex.score(gt, v1)}
        print(t, {k: round(v["spell_f1_mean"], 4) for k, v in res[t].items()},
              "D-F1", {k: round(v["D_f1"], 3) for k, v in res[t].items()}, flush=True)
    json.dump(res, open(os.path.join(ex.OUT, "broadcast_cue.json"), "w"), indent=1)
    for k in ("without", "with_cue"):
        print(f"{k:9s} spellF1 {np.mean([res[t][k]['spell_f1_mean'] for t in res]):.4f}  "
              f"D-F1 {np.mean([res[t][k]['D_f1'] for t in res]):.4f}  "
              f"macroF1 {np.mean([res[t][k]['macro_f1'] for t in res]):.4f}")
    print("BROADCAST_DONE")
