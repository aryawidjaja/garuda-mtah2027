"""Loader for the external (non-competition) tracking datasets built by scripts/build_external.py.

Each match lives in external/processed/<match_id>.npz plus <match_id>.json (metadata).

Arrays (T = frames, P = every player who appears in the match, subs included):
    frame          (T,)     int64   native frame number of the source (not contiguous across halves)
    period         (T,)     int8    1 or 2
    ball_xy        (T,2)    float32 metres, origin pitch centre, x along the length; NaN = missing
    ball_z         (T,)     float32 metres (NaN where the source has none)
    players_xy     (T,P,2)  float32 metres, same frame as ball_xy; NaN = player not on pitch / not tracked
    players_team   (P,)     int8    0 = home, 1 = away
    players_is_gk  (P,)     bool
    player_ids     (P,)     str     source ids; internal only, anonymise before sharing
    label          (T,)     int8    0 = H (home possession, ball alive), 1 = A (away, alive), 2 = D (dead)
    label_is_proxy ()       bool    True when label was inferred from events, not given per frame
Source-specific raw columns are kept too (e.g. ball_status, ball_possession_raw for IDSSE);
see the metadata "arrays" entry for what each match carries.
"""
import json
import os

import numpy as np

ROOT = os.environ.get(
    "GARUDA_EXT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external"))
PROCESSED = os.path.join(ROOT, "processed")
H, A, D = 0, 1, 2
LABEL_CHARS = np.array(list("HAD"))


def list_matches(source=None):
    ids = sorted(f[:-4] for f in os.listdir(PROCESSED) if f.endswith(".npz"))
    return [m for m in ids if source is None or m.startswith(source + "_")]


def load_match(match_id):
    """dict of the arrays above plus 'meta' (the sidecar JSON)."""
    with np.load(os.path.join(PROCESSED, match_id + ".npz"), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    with open(os.path.join(PROCESSED, match_id + ".json")) as f:
        out["meta"] = json.load(f)
    return out


def runs(x):
    """(start, end_exclusive, value) of maximal constant runs of a 1-D array."""
    x = np.asarray(x)
    if x.size == 0:
        return []
    cut = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, x.size]
    return list(zip(starts.tolist(), ends.tolist(), x[starts].tolist()))


if __name__ == "__main__":
    assert runs([0, 0, 2, 2, 0, 1]) == [(0, 2, 0), (2, 4, 2), (4, 5, 0), (5, 6, 1)]
    for m in list_matches():
        d = load_match(m)
        T = d["label"].size
        assert d["ball_xy"].shape == (T, 2) and d["players_xy"].shape[0] == T
        assert d["players_xy"].shape[1] == d["players_team"].size
        assert set(np.unique(d["label"])) <= {H, A, D}
        print(m, T, np.bincount(d["label"], minlength=3) / T)
