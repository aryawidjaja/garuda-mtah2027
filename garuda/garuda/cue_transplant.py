"""TV-feed cue for DFL matches, transplanted from real SkillCorner camera behaviour.

SkillCorner tracking comes from its own broadcast, so there the cue is redundant with the tracking
gaps. BAR-ATH's tactical tracking is independent of its broadcast, and so is DFL's (no TV feed).
Each DFL run (live stretch or stoppage) gets the "live / not live" pattern of a random SkillCorner
run of the same kind and similar length, time-warped to fit: cue statistics are SkillCorner's,
the tracking is independent of them.
"""

import numpy as np

from . import broadcast, ext_data


def _runs(alive):
    m = np.concatenate(([-1], alive.astype(np.int8), [-1]))
    e = np.flatnonzero(np.diff(m) != 0)
    return [(int(a), int(b), bool(alive[a])) for a, b in zip(e[:-1], e[1:])]


def build_library(mids):
    """{True: [live-run patterns], False: [dead-run patterns]} at 25 Hz (from 10 Hz, nearest)."""
    lib = {True: [], False: []}
    for m in mids:
        d = ext_data.load_match(m)
        live = broadcast.skillcorner_live(m)
        for p in np.unique(d["period"]):
            idx = np.flatnonzero(d["period"] == p)
            alive = d["label"][idx] != 2
            for a, b, al in _runs(alive):
                seg = live[idx[a:b]]
                if b - a >= 5 and not np.isnan(seg).any():
                    lib[al].append(np.repeat(seg, 2 + (np.arange(len(seg)) % 2)))   # 10 -> 25 Hz
    return lib


def transplant(label25, lib, rng):
    """(T,) bc_live for a 25 Hz label array (2 = dead)."""
    out = np.empty(len(label25), np.float32)
    lens = {k: np.array([len(x) for x in v]) for k, v in lib.items()}
    for a, b, al in _runs(label25 != 2):
        n = b - a
        cand = np.flatnonzero((lens[al] > 0.6 * n) & (lens[al] < 1.6 * n))
        pick = lib[al][rng.choice(cand) if len(cand) else int(np.argmin(np.abs(lens[al] - n)))]
        out[a:b] = pick[np.minimum((np.arange(n) * len(pick) / n).astype(int), len(pick) - 1)]
    return out


if __name__ == "__main__":
    lib = {True: [np.ones(100, np.float32), np.r_[np.ones(80), np.zeros(20)].astype(np.float32)],
           False: [np.r_[np.ones(10), np.zeros(90)].astype(np.float32)]}
    lab = np.r_[np.zeros(200), np.full(100, 2), np.ones(150)].astype(np.int8)
    cue = transplant(lab, lib, np.random.default_rng(0))
    assert len(cue) == len(lab) and cue[200:300].mean() < 0.3 and cue[:200].mean() > 0.7
    print("ok")
