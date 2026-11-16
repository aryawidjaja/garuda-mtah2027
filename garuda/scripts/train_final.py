"""Train the shipped model with the frozen configuration on every open source and write
garuda/model/{lgbm.txt, gru.pt, config.json}. Training need not be bit-reproducible (official issue #11);
inference with the committed weights is.

    python garuda/scripts/train_final.py
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
import experiment_broadcast as eb  # noqa: E402
import experiment_transplant as et  # noqa: E402
import exp_weight as ew, exp_weight_cue as ewc  # noqa: E402
from garuda import broadcast, cue_transplant, ext_data, features, seq  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)   # GRU training: two threads, as in its experiments

USE_CUE = True   # TV-feed "not shown live" cue (experiments F and G); False reproduces model v1
METRICA_W = 5    # Metrica rows weighted 5, plus 3 degraded draws each; no SkillCorner label rows.
                 # 0 reproduces model v2 (SkillCorner native rows + Metrica clean, all weight 1)

MODEL_DIR = os.path.join(os.path.dirname(HERE), "model")


def rows(lib):
    """Training rows of every open source: (feature names, Xs, ys, per-block weights)."""
    dfl = ext_data.list_matches("idsse")
    sc = ext_data.list_matches("skillcorner")
    Xs, ys = [], []
    for i, m in enumerate(dfl):
        for k in range(ex.K_DRAWS):
            seed = 50_000 + 100 * i + k
            names, X, y, per, _, sw = ex.match_features(m, seed, True)
            Xs.append(et.with_cue(X, per, sw, et.cue_blocks(m, seed, lib)) if USE_CUE else X); ys.append(y)
    n_dfl = len(Xs)
    for m in (sc if not METRICA_W else []):
        X0, X1, y, _, _ = es.sc_features(m)
        if USE_CUE:
            b = eb.sc_bc(m)
            X0, X1 = np.hstack([X0, b]), np.hstack([X1, b])
        Xs += [X0, X1]; ys += [y, np.where(y == 2, 2, 1 - y)]
    for i, m in enumerate(ext_data.list_matches("metrica")):
        X0, X1, y, _, _ = em.raw_features(m)
        if USE_CUE:
            X0, X1 = np.hstack([X0, eb.nan_block(len(X0))]), np.hstack([X1, eb.nan_block(len(X1))])
        Xs += [X0, X1]; ys += [y, np.where(y == 2, 2, 1 - y)]
        for k in range(ex.K_DRAWS if METRICA_W else 0):
            seed = 50_500 + 10 * i + k
            X0, X1, y, per, _ = ew.met_degraded(m, seed)
            if USE_CUE:
                X0, X1 = ewc.met_cue(m, seed, lib, X0, X1, per)
            Xs += [X0, X1]; ys += [y, np.where(y == 2, 2, 1 - y)]
    if USE_CUE:
        names = list(names) + broadcast.features(np.zeros(10, np.float32))[0]
    w = max(METRICA_W, 1)
    return names, Xs, ys, [1] * n_dfl + [w] * (len(Xs) - n_dfl)


def main():
    cfg = json.load(open(os.path.join(ex.OUT, "frozen.json")))
    dfl = ext_data.list_matches("idsse")
    lib = cue_transplant.build_library(ext_data.list_matches("skillcorner")) if USE_CUE else None
    names, Xs, ys, ws = rows(lib)
    clf, prior = ew.fit(Xs, ys, ws) if METRICA_W else em.fit(Xs, ys)
    print("lightgbm fitted", flush=True)
    gru = seq.train(Xs, ys, ws)   # v4: its probabilities are averaged with LightGBM's
    h, alpha = ex.hsmm_from_labels(dfl, tuple(cfg["min_s"]), cfg["alpha"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    clf.booster_.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    seq.save(gru, os.path.join(MODEL_DIR, "gru.pt"))
    json.dump({
        "feature_names": list(names), "prior": prior.tolist(), "gru": "gru.pt", "alpha": alpha,
        "min_dur": h.min_dur.tolist(), "stay": h.stay_probs, "switch": h.switch_matrix,
        "mbr": cfg["mbr"], "broadcast_cue": USE_CUE, "feature_version": features.VERSION, "tag_beta": cfg["tag_beta"], "tag_gamma": cfg["tag_gamma"], "seed": 0,
        "target": ex.TARGET, "metrica_weight": METRICA_W,
        "trained_on": dfl + ([] if METRICA_W else sc) + ext_data.list_matches("metrica"),
        "cue_library": "skillcorner camera behaviour only" if METRICA_W else "skillcorner",
    }, open(os.path.join(MODEL_DIR, "config.json"), "w"), indent=1)
    print("wrote", MODEL_DIR)


if __name__ == "__main__":
    main()
