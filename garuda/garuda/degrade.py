"""Make clean optical tracking (other matches) look like automatic single-camera tracking.

Domain randomisation: each training draw samples its own corruption parameters, so the model sees
a family of trackers rather than one guess. Ranges are wide and centred on published facts about
the target tracker: DATA.md, ball visible on ~57-59% of frames and ~20.5 of 22 players tracked
during play; danieltyukov's public technical report (same tracker), gaps mostly >= 3 s (shorter
ones look bridged by the provider) and the ball missing far more inside stoppages than in open play.
DFL quirk: the optical feed freezes the ball during stoppages and jumps it to the restart spot,
which would teach "still ball = dead", a cue the target tracker does not show. Stoppage balls are
rebuilt as a roll-out, then carried at walking pace.
Mechanisms (why an automatic tracker loses things):
  occlusion gaps   ball missing more often when crowded; lognormal gap lengths
  aerial ball      lost when high (if the source has ball height)
  dead ball        whole stoppages often untracked (ball out of view, carried, replaced)
  false ball       short spurious detections
  players          intermittent dropout + out-of-view far from the action + duplicates
  noise            correlated position jitter
"""

import numpy as np


def sample_params(rng):
    return {
        "miss_target": rng.uniform(0.08, 0.30),     # ball missing share of alive frames
        "occ_med_s": rng.uniform(2.0, 6.0),          # median occlusion gap
        "bridge_s": rng.uniform(0.5, 3.0),           # shorter gaps are interpolated, not missing
        "cam_back": rng.uniform(15.0, 45.0),         # camera behind the far touchline (aerial)
        "cam_h": rng.uniform(12.0, 30.0),
        "crowd_k": rng.uniform(0.0, 1.5),            # hazard multiplier per player within 3 m
        "air_p": rng.uniform(0.3, 0.9),              # P(lost | ball above 2 m)
        "dead_p": rng.uniform(0.5, 0.95),            # P(stoppage mostly untracked)
        "false_rate": rng.uniform(0.0, 0.004),       # spurious-ball segment starts per frame
        "pl_drop_rate": rng.uniform(0.0005, 0.004),  # player dropout starts per frame
        "pl_drop_s": rng.uniform(0.5, 4.0),
        "view_half": rng.uniform(32.0, 70.0),        # metres either side of the camera aim
        "view_p": rng.uniform(0.3, 1.0),             # P(out-of-view player is dropped)
        "dup_rate": rng.uniform(0.0, 0.0005),
        "pl_sigma": rng.uniform(0.05, 0.35),
        "ball_sigma": rng.uniform(0.1, 0.8),
    }


def _segments(mask):
    """[(start, stop)] runs of True."""
    m = np.concatenate(([False], mask, [False]))
    e = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(e[::2], e[1::2]))


def _rebuild_dead_ball(ball, label, rng):
    """In place: replace the source's frozen stoppage ball by a roll-out then a walk to the restart."""
    T = len(ball)
    for s, e in _segments(label == 2):
        if s < 6 or e >= T:
            continue
        v = (ball[s - 1] - ball[s - 6]) / 5
        if not np.all(np.isfinite(v)):
            v = np.zeros(2, np.float32)
        roll = min(e - s, int(rng.uniform(0.3, 2.0) * 25))
        k = np.arange(1, roll + 1, dtype=np.float32)[:, None]
        decel = k - k ** 2 / (2 * max(roll, 1))           # linear deceleration to rest
        ball[s:s + roll] = ball[s - 1] + v * decel
        start = ball[s + roll - 1] if roll else ball[s - 1]
        end = ball[e] if np.all(np.isfinite(ball[e])) else start
        n = e - (s + roll)
        if n > 0:
            w = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            ball[s + roll:e] = start + w * (end - start)


def _bridge(ball, max_len):
    """In place: linearly interpolate NaN gaps shorter than max_len (provider bridging)."""
    miss = np.isnan(ball[:, 0])
    for s, e in _segments(miss):
        if 0 < s and e < len(ball) and e - s < max_len:
            w = np.linspace(0, 1, e - s + 2, dtype=np.float32)[1:-1, None]
            ball[s:e] = ball[s - 1] + w * (ball[e] - ball[s - 1])


def _ar_noise(rng, shape, sigma, rho=0.9):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), shape).astype(np.float32)
    for t in range(1, shape[0]):
        e[t] += rho * e[t - 1]
    return e


def _gap_mask(rng, T, hazard, med_frames):
    miss = np.zeros(T, bool)
    starts = np.flatnonzero(rng.random(T) < hazard)
    lens = np.maximum(1, rng.lognormal(np.log(med_frames), 0.9, len(starts))).astype(int)
    for s, n in zip(starts, lens):
        miss[s:s + n] = True
    return miss


def degrade(d, label, p, rng):
    """Return a corrupted copy of d. label (T,) 0/1/2 is used only to place dead-ball loss."""
    T = len(d["ball_xy"])
    ball = d["ball_xy"].astype(np.float32).copy()
    pl = d["players_xy"].astype(np.float32).copy()
    team = d["players_team"]
    z = d.get("ball_z")
    alive = label != 2
    crowd = np.nansum(np.hypot(pl[..., 0] - ball[:, None, 0], pl[..., 1] - ball[:, None, 1]) < 3, 1)

    miss = np.zeros(T, bool)
    _rebuild_dead_ball(ball, label, rng)
    if z is not None:
        zz = np.nan_to_num(z).astype(np.float32)
        W = d["pitch"][1]
        cam = np.array([0.0, W / 2 + p["cam_back"]], np.float32)
        lift = np.clip(zz / np.maximum(p["cam_h"] - zz, 1.0), 0, 2)[:, None]
        ball = ball + lift * (ball - cam)          # a high ball lands further along the camera ray
        miss |= (zz > 2.0) & (rng.random(T) < p["air_p"])
    for s, e in _segments(~alive):
        if rng.random() < p["dead_p"]:
            a = s + int(rng.uniform(0, 0.3) * (e - s))
            b = e - int(rng.uniform(0, 0.3) * (e - s))
            miss[a:b] = True
    # occlusion: scale the base hazard until the alive-frame missing share hits the target
    med = p["occ_med_s"] * 25
    weight = 1.0 + p["crowd_k"] * crowd
    lo, hi = 0.0, 3.0
    for _ in range(18):
        h = (lo + hi) / 2
        g = _gap_mask(np.random.default_rng(int(rng.integers(1 << 30))), T, h * weight / med, med)
        share = (miss | g)[alive].mean() if alive.any() else 0
        lo, hi = (h, hi) if share < p["miss_target"] else (lo, h)
    miss |= _gap_mask(rng, T, ((lo + hi) / 2) * weight / med, med)
    ball[miss] = np.nan
    _bridge(ball, int(p["bridge_s"] * 25))
    ball += _ar_noise(rng, ball.shape, p["ball_sigma"])

    # spurious detections: a short run at a random player's position
    for s in np.flatnonzero(rng.random(T) < p["false_rate"]):
        k = rng.integers(pl.shape[1])
        n = int(rng.integers(2, 15))
        ball[s:s + n] = pl[s:s + n, k] + rng.normal(0, 1.0, 2)

    # players: dropout, out of camera view, duplicates, noise
    P = pl.shape[1]
    drop = np.zeros((T, P), bool)
    for k in range(P):
        drop[:, k] = _gap_mask(rng, T, p["pl_drop_rate"], p["pl_drop_s"] * 25)
    aim = np.copy(d["ball_xy"][:, 0])
    ok = ~np.isnan(aim)
    aim = np.interp(np.arange(T), np.flatnonzero(ok), aim[ok]) if ok.any() else np.zeros(T)
    a = np.empty(T, np.float32)
    a[0] = aim[0]
    for t in range(1, T):          # a camera operator smoothly follows the play (~1 s lag)
        a[t] = a[t - 1] + 0.04 * (aim[t] - a[t - 1])
    far = np.abs(pl[..., 0] - a[:, None]) > p["view_half"]
    drop |= far & (rng.random((T, P)) < p["view_p"])
    pl[drop] = np.nan
    pl += _ar_noise(rng, pl.shape, p["pl_sigma"])
    dup = rng.random(T) < p["dup_rate"]
    extra = np.full((T, 1, 2), np.nan, np.float32)
    if dup.any():
        k = rng.integers(P)
        for s in np.flatnonzero(dup):
            n = int(rng.integers(10, 100))
            extra[s:s + n, 0] = pl[s:s + n, k] + 0.8
    out = dict(d)
    out["ball_xy"] = ball
    out["players_xy"] = np.concatenate([pl, extra], 1)
    out["players_team"] = np.concatenate([team, [team[k] if dup.any() else 0]]).astype(np.int8)
    out.pop("ball_z", None)
    # the simulated camera (centred metres + height); features use it for line-of-sight geometry
    out["cam"] = np.array([0.0, d["pitch"][1] / 2 + p["cam_back"], p["cam_h"]], np.float32)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, P = 3000, 22
    d = {"ball_xy": rng.normal(0, 10, (T, 2)).astype(np.float32),
         "players_xy": rng.normal(0, 15, (T, P, 2)).astype(np.float32),
         "players_team": np.repeat([0, 1], 11).astype(np.int8), "pitch": (105, 68)}
    lab = np.zeros(T, int)
    lab[1000:1300] = 2
    p = sample_params(rng)
    o = degrade(d, lab, p, rng)
    share = np.isnan(o["ball_xy"][:, 0])[lab != 2].mean()
    print("target %.2f got %.2f, players present %.1f" % (
        p["miss_target"], share, (~np.isnan(o["players_xy"][..., 0])).sum(1).mean()))
    assert abs(share - p["miss_target"]) < 0.12
    print("ok")
