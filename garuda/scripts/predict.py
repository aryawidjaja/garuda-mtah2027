"""BAR-ATH -> Garuda.csv. Deterministic CPU inference with the committed model.

    export MTAH_DATA_DIR=/path/to/data      # parent of tactical/
    python garuda/scripts/predict.py --out outputs/private/Garuda.csv

Prints no match data: only the output path and the validator verdict.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(ROOT, "official-repo"))
sys.path.insert(0, os.path.join(ROOT, "official-repo", "evaluation"))
from garuda import atd, broadcast, decode, features, kickoff, tags  # noqa: E402
import games  # noqa: E402

MODEL_DIR = os.path.join(PKG, "model")
# Smart Tagging team strings as documented in DATA.md (home is Barcelona).
TAG_TEAM = {"barcelona": 0, "atletic club": 1}


def load_model():
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    features.VERSION = int(cfg.get("feature_version", 1))   # features exactly as the model was trained
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm.txt"))
    return booster, cfg


def decode_half(booster, cfg, sub, sp_windows, goals, kos, live25=None):
    names, X = features.build(sub)
    _, Xs = features.build(sub, swap=True)
    if cfg.get("broadcast_cue"):
        if live25 is None:
            live25 = np.full(len(sub["ball_xy"]), np.nan, np.float32)
        bn, B = broadcast.features(live25)
        names, X, Xs = names + bn, np.hstack([X, B]), np.hstack([Xs, B])
    assert names == cfg["feature_names"], "feature list changed since training"
    p = (booster.predict(X) + booster.predict(Xs)[:, [1, 0, 2]]) / 2
    decode_half.last_proba = p      # per-frame probabilities, saved next to the CSV (pred_dump.npz)
    e = decode.emissions(p, cfg["prior"], cfg["alpha"])
    n5 = len(e)
    if cfg.get("tag_beta", 0) or cfg.get("tag_gamma", 0):
        e = e + tags.bonus_from_windows(sp_windows, n5, cfg["tag_beta"], cfg["tag_gamma"], None)
        e = e + tags.goal_bonus(goals, kos, n5, cfg["tag_beta"])
    h = decode.HSMM(cfg["min_dur"], cfg["stay"], np.array(cfg["switch"]))
    vit = h.viterbi(e)
    if cfg.get("mbr", True):
        smp = h.sample_paths(e, cfg.get("n_samples", 24), seed=cfg.get("seed", 0))
        cands = [vit] + list(smp)
        vit = cands[decode.mbr_select(cands, smp)[0]]
    n = len(sub["ball_xy"])
    lab = np.repeat(vit, features.STEP)
    return np.concatenate([lab, np.full(max(0, n - len(lab)), lab[-1])])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "private", "Garuda.csv"))
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    meta, raw = games.require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    n = games.source_frames()
    cache = os.path.join(os.path.dirname(args.out), "atd_cache.npz")
    d = atd.read_atd(meta, raw, n, games.GAMES[games.GAME]["pitch"], cache=cache)
    try:
        tg = atd.read_tags(games.require(["smart_tagging"], data_dir=args.data_dir)[0])
    except Exception:  # noqa: BLE001 -- tags are optional evidence; never block the submission
        tg = []
    booster, cfg = load_model()
    cam = None
    if features.VERSION >= 3:
        try:
            from garuda import camera
            Hn = atd.read_homographies(games.require(["homography"], data_dir=args.data_dir)[0], n)
            cam, _ = camera.match_camera(Hn, *d["pitch"])
        except Exception:  # noqa: BLE001 -- unknown camera -> line-of-sight features stay NaN
            cam = None
    bc = None
    if cfg.get("broadcast_cue"):
        try:  # optional input: any failure leaves the cue unknown (NaN), never breaks the run
            bmeta, braw, bhom = games.require(["atd_meta", "atd_raw", "homography"],
                                              data_dir=args.data_dir, source="broadcast")
            nb = games.source_frames("broadcast")
            bd = atd.read_atd(bmeta, braw, nb, games.GAMES[games.GAME]["pitch"],
                              cache=os.path.join(os.path.dirname(args.out), "atd_bc_cache.npz"))
            bc = (bd["ball_xy"], atd.read_valid(bhom, nb), bd["players_xy"])
        except Exception:  # noqa: BLE001
            bc = None
    out = np.full(n, "X", dtype="<U1")
    proba25 = np.full((n, 3), np.nan, np.float32)
    bc25 = np.full(n, np.nan, np.float32)
    kicks = []
    for half, first, last in games.half_video_blocks(games.GAME):
        sl = slice(first, last + 1)
        sub = {"ball_xy": d["ball_xy"][sl], "players_xy": d["players_xy"][sl],
               "players_team": d["players_team"], "pitch": d["pitch"], "cam": cam}
        sp, goals, kos = [], [], []
        for t in tg:
            if first <= t["start"] <= last:
                s, e = t["start"] - first, t["end"] - first
                if t["code"] == "set pieces":
                    sp.append((s, e, TAG_TEAM.get(t["team"])))
                    if t["type"].startswith("kick off"):
                        kos.append(s)
                elif t["code"] == "goals":
                    goals.append(e)
        live = None
        if bc is not None:
            live, _ = broadcast.tactical_live(d["ball_xy"], bc[0], bc[1], first, last,
                                              tac_pl=d["players_xy"], bc_pl=bc[2])
        if cfg.get("broadcast_cue"):   # one bit per half, nothing about the match itself
            print(f"broadcast cue half {half}:", "on" if live is not None and not np.isnan(live).all() else "off")
        lab = decode_half(booster, cfg, sub, sp, goals, sorted(kos), live)
        out[sl] = np.array(list("HAD"))[lab]
        kick = kickoff.apply(out, tg, d["ball_xy"], first, last)   # no live play before a detected kickoff
        kicks.append(-1 if kick is None else int(kick))
        p5 = decode_half.last_proba
        proba25[sl] = np.repeat(p5, features.STEP, axis=0)[:last - first + 1] if len(p5) * features.STEP >= last - first + 1 \
            else np.vstack([np.repeat(p5, features.STEP, axis=0), np.repeat(p5[-1:], last - first + 1 - len(p5) * features.STEP, axis=0)])
        if live is not None:
            bc25[sl] = live
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{c}\n" for i, c in enumerate(out))
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(args.out)), "pred_dump.npz"),
                        proba=proba25.astype(np.float16), bc_live=bc25.astype(np.float16),
                        kickoffs=np.array(kicks, np.int64))
    import possession_metrics as pm
    reason = pm.validate_submission(pm.read_label_csv(args.out), n)
    print(args.out, "VALID" if reason is None else "INVALID (format)")


if __name__ == "__main__":
    main()
