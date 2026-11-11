"""Extra training sources in the common 25 Hz dict layout.

SkillCorner open data (A-League 2024/25, broadcast tracking, 10 fps, MIT):
  dead  = outside SkillCorner's phases of play (proxy, from the data build)
  team  = SkillCorner's per-frame possession group, carried forward through nulls inside a live
          run; a run that opens with null takes the phase's team.
  Resampled 10 -> 25 Hz by linear interpolation (NaN if either neighbour is NaN; no invention).
Metrica sample games: already 25 Hz, used as delivered (event-proxy labels).
"""

import numpy as np

from . import ext_data


def _sk_labels(d):
    lab = d["label"].astype(np.int8).copy()
    pg = d["possession_group"]
    alive = lab != 2
    edges = np.flatnonzero(np.diff(np.concatenate(([0], alive.astype(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        cur = int(lab[a])
        for t in range(a, b):
            if pg[t] >= 0:
                cur = int(pg[t])
            lab[t] = cur
    return lab


def _resample(x, n_out, ratio):
    t = np.arange(n_out) / ratio
    i = np.minimum(np.floor(t).astype(int), len(x) - 2)
    w = (t - i).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return (1 - w) * x[i] + w * x[i + 1]


def skillcorner_periods(mid):
    """[dict] one per period at 25 Hz."""
    d = ext_data.load_match(mid)
    lab = _sk_labels(d)
    out = []
    for p in np.unique(d["period"]):
        idx = np.flatnonzero(d["period"] == p)
        n10 = len(idx)
        n25 = int((n10 - 1) * 2.5)
        if n25 < 100:
            continue
        nearest = np.minimum(np.round(np.arange(n25) / 2.5).astype(int), n10 - 1)
        out.append({
            "ball_xy": _resample(d["ball_xy"][idx], n25, 2.5),
            "players_xy": _resample(d["players_xy"][idx], n25, 2.5),
            "players_team": d["players_team"],
            "label": lab[idx][nearest],
            "pitch": (d["meta"]["pitch_length"], d["meta"]["pitch_width"]),
        })
    return out


if __name__ == "__main__":
    d = {"label": np.array([2, 0, 0, 0, 0, 2, 1, 1], np.int8),
         "possession_group": np.array([-1, -1, 1, -1, 0, -1, -1, 0], np.int8)}
    assert list(_sk_labels(d)) == [2, 0, 1, 1, 0, 2, 1, 0]
    x = np.array([[0.0], [10.0], [np.nan]], np.float32)
    r = _resample(x, 5, 2.5)
    assert np.allclose(r[:3, 0], [0, 4, 8]) and np.isnan(r[3, 0])
    print("ok")
