"""TV-feed "not shown live" cue, bc_live in {0, 1} per 25 Hz frame (NaN = unknown).

A broadcast shows the pitch during live play and cuts to replays and close-ups mostly while the
ball is dead, so a broadcast frame without pitch geometry suggests a dead ball.
Training (other matches): SkillCorner's per-frame camera footprint (image_corners_projection),
null when the image shows no pitch. Competition: the broadcast homography `valid` flag, mapped to
tactical frames by a constant per-half offset found by matching the two tracking files (no labels).
"""

import json
import os

import numpy as np

from . import ext_data


def skillcorner_live(mid):
    """(T,) float32 aligned to the processed SkillCorner frames: 1 pitch visible, 0 not, NaN unknown."""
    d = ext_data.load_match(mid)
    sid = mid.split("_", 1)[1]
    raw = os.path.join(ext_data.ROOT, "raw", "skillcorner_lfs")
    fn = [f for f in os.listdir(raw) if f.startswith(sid) and "tracking" in f]
    if not fn:
        return np.full(len(d["frame"]), np.nan, np.float32)
    live = {}
    with open(os.path.join(raw, fn[0])) as fh:
        for line in fh:
            r = json.loads(line)
            c = r.get("image_corners_projection") or {}
            live[r["frame"]] = 1.0 if c and all(v is not None for v in c.values()) else 0.0
    return np.array([live.get(int(f), np.nan) for f in d["frame"]], np.float32)


def _agree(a, b, tol):
    both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    if both.sum() < 200:
        return 0.0
    return float((np.hypot(*(a[both] - b[both]).T) < tol).mean())


def best_offset(tac_ball, bc_ball, lo=None, hi=None, coarse=5, sub=25, tol=2.0):
    """Constant offset k (bc_frame = tac_frame - k) maximising ball agreement within tol metres.
    Coarse pass: every `coarse`-th offset on every `sub`-th tactical frame; then every offset
    within +/-coarse on all frames. Returns (k, agreement share among co-observed frames)."""
    lo = -len(bc_ball) if lo is None else lo          # every possible overlap of the two files
    hi = len(tac_ball) if hi is None else hi
    t = np.arange(0, len(tac_ball), sub)
    best = (0, 0.0)
    for k in range(lo, hi, coarse):
        j = t - k
        ok = (j >= 0) & (j < len(bc_ball))
        if ok.sum() < 200:
            continue
        a = _agree(tac_ball[t[ok]], bc_ball[j[ok]], tol)
        if a > best[1]:
            best = (k, a)
    k0, best_med = best[0], np.inf
    for k in range(k0 - coarse, k0 + coarse + 1):     # refine: smallest median distance
        j = np.arange(len(tac_ball)) - k
        ok = (j >= 0) & (j < len(bc_ball))
        a, b = tac_ball[ok], bc_ball[j[ok]]
        both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        if both.sum() < 200:
            continue
        med = float(np.median(np.hypot(*(a[both] - b[both]).T)))
        if med < best_med:
            best_med, best = med, (k, _agree(a, b, tol))
    return best


def _players_agree(tac_pl, bc_pl, tol=3.0, min_n=4):
    """Per frame: 1 if the broadcast's tracked players sit where the tactical ones are (median
    nearest distance < tol), 0 if clearly elsewhere (a replay or another moment), NaN if too few."""
    out = np.full(len(tac_pl), np.nan, np.float32)
    for t in range(len(tac_pl)):
        b = bc_pl[t][~np.isnan(bc_pl[t, :, 0])]
        a = tac_pl[t][~np.isnan(tac_pl[t, :, 0])]
        if len(b) >= min_n and len(a) >= min_n:
            dd = np.hypot(b[:, None, 0] - a[None, :, 0], b[:, None, 1] - a[None, :, 1]).min(1)
            out[t] = float(np.median(dd) < tol)
    return out


def tactical_live(tac_ball, bc_ball, bc_valid, first, last, min_agree=0.5, tac_pl=None, bc_pl=None):
    """bc_live for tactical frames [first, last] of one half; all-NaN if the alignment is not
    convincing, so the model falls back to its other evidence. With player arrays, a valid
    broadcast frame whose players do not match the tactical ones (a replay) counts as 0."""
    n = last - first + 1
    k, agree = best_offset(tac_ball[first:last + 1], bc_ball)
    out = np.full(n, np.nan, np.float32)
    if agree < min_agree:
        return out, (k, agree)
    idx = np.arange(first, last + 1) - first - k
    ok = (idx >= 0) & (idx < len(bc_valid))
    out[ok] = bc_valid[idx[ok]].astype(np.float32)
    if tac_pl is not None and bc_pl is not None:
        j = np.flatnonzero(ok & (out == 1))
        if len(j):
            match = _players_agree(tac_pl[first + j], bc_pl[idx[j]])
            out[j[match == 0]] = 0.0
    return out, (k, agree)


def features(live25, step=5):
    """5 Hz cue columns: mean over the block and over past/future windows."""
    from .features import _pool, _roll
    p = _pool(live25.astype(np.float32))
    cols = {"bc_live": p}
    for name, (lo, hi) in {"c1": (-1, 1), "c3": (-3, 3), "c8": (-8, 8), "p3": (-3, 0),
                           "f3": (0, 3), "p8": (-8, 0), "f8": (0, 8)}.items():
        cols[f"bc_live_{name}"] = _roll(p, lo, hi)
    names = sorted(cols)
    return names, np.column_stack([cols[k] for k in names]).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tt = np.arange(20000) / 25.0                  # a smoothly moving ball, like real tracking
    tac = np.column_stack([30 * np.sin(tt / 7) + 5 * np.sin(tt * 1.3), 20 * np.cos(tt / 5)]).astype(np.float32)
    tac[rng.random(20000) < 0.4] = np.nan
    bc = np.full((15000, 2), np.nan, np.float32)
    bc[:] = tac[3000:18000]                       # bc frame j = tactical frame j + 3000  -> k = 3000
    bc[rng.random(15000) < 0.3] = np.nan
    k, agree = best_offset(tac, bc)
    assert k == 3000 and agree > 0.95, (k, agree)
    bc2 = np.full((30000, 2), np.nan, np.float32)  # tactical half starts deep inside the bc timeline
    bc2[12000:27000] = tac[:15000]                # bc frame = tactical frame + 12000 -> k = -12000
    k2, agree2 = best_offset(tac[:15000], bc2)
    assert k2 == -12000 and agree2 > 0.95, (k2, agree2)
    live, _ = tactical_live(tac, bc, np.ones(15000), 0, 19999)
    assert np.isnan(live[:3000]).all() and (live[3000:18000] == 1).all()
    tp = rng.normal(0, 20, (50, 10, 2)).astype(np.float32)
    m = _players_agree(tp, np.concatenate([tp[:25], tp[25:] + 30]))
    assert (m[:25] == 1).all() and (m[25:] == 0).all()
    n, X = features(np.r_[np.ones(500), np.zeros(500)].astype(np.float32))
    assert X.shape == (200, 8)
    print("ok")
