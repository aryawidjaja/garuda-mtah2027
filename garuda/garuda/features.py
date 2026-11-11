"""Identity-free per-frame features, pooled to 5 Hz, with past/future context windows.

Input dict (both sources): ball_xy (T,2) metres centred, NaN = not tracked; players_xy (T,P,2)
metres, NaN = absent; players_team (P,) 0 home / 1 away; pitch (L, W); fps 25.
Features never use track identity beyond a 0.2 s central difference for speed, because the
competition tracker swaps and fragments slots.

Teams are referred to as "a" (the team whose possession is label 0) and "b". Calling build() with
swap=True exchanges the teams and mirrors x, which is the H<->A augmentation.
"""

import numpy as np

STEP = 5           # 25 Hz -> 5 Hz
DT = 1.0 / 25
# Multi-scale context in 25 Hz frames: c1 = +/-0.2 s, c3 = +/-0.6 s, c8 = +/-1.6 s, c20 = +/-4 s,
# p/f = past-only / future-only. (Lang et al. 2022 used up to +/-80 s; longer windows are untested here.)
WINDOWS = {"c1": (-5, 5), "c3": (-15, 15), "c8": (-40, 40), "c20": (-100, 100),
           "p3": (-15, 0), "f3": (0, 15), "p8": (-40, 0), "f8": (0, 40)}
MAXGAP = 250       # frames; caps "time since / until ball seen"
VERSION = 1        # default = the shipped model's features. Experiments set 2 (+ control history,
                   # ball heading, long windows) or 3 (+ line of sight) explicitly. predict.py uses
                   # the version recorded in the model's config.
HEADER = 2.5       # metres: the highest a player can control the ball
CTRL_CAP = 750     # frames (30 s) cap for time since / until clear control


def _nanmean(x, axis):
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=axis)


def _central_speed(xy):
    """(T,...,2) -> speed (T,...) m/s from a 0.2 s central difference, NaN where undefined."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    d = (xy[4:] - xy[:-4]) / (4 * DT)
    v[2:-2] = np.hypot(d[..., 0], d[..., 1])
    v[v > 13] = np.nan   # identity switch, not a sprint
    return v


def _ffill_idx(mask):
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_idx(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def attack_sign(players_xy, team):
    """+1 if team 0 (home) attacks +x in this period: the defending block sits nearer its own goal."""
    c0 = np.nanmean(players_xy[:, team == 0, 0])
    c1 = np.nanmean(players_xy[:, team == 1, 0])
    return 1.0 if c0 < c1 else -1.0


def frame_features(d, swap=False):
    L, W = d["pitch"]
    ball = d["ball_xy"].astype(np.float32).copy()
    pl = d["players_xy"].astype(np.float32).copy()
    team = d["players_team"].astype(np.int8)
    if swap:
        team = 1 - team
    sgn = attack_sign(pl, team)            # orient: team a attacks +x
    ball[:, 0] *= sgn
    pl[..., 0] *= sgn
    cam = None if d.get("cam") is None else np.array([d["cam"][0] * sgn, d["cam"][1], d["cam"][2]], np.float32)
    T = len(ball)
    f = {}
    present = ~np.isnan(pl[..., 0])
    spd = _central_speed(pl)
    for k, name in ((0, "a"), (1, "b")):
        m = team == k
        p = pl[:, m]
        pres = present[:, m]
        f[f"n_{name}"] = pres.sum(1).astype(np.float32)
        f[f"cx_{name}"] = _nanmean(p[..., 0], 1)
        f[f"cy_{name}"] = np.abs(_nanmean(p[..., 1], 1))
        with np.errstate(invalid="ignore"):
            f[f"sx_{name}"] = np.nanstd(p[..., 0], 1)
            f[f"sy_{name}"] = np.nanstd(p[..., 1], 1)
            f[f"maxx_{name}"] = np.nanmax(np.where(pres, p[..., 0], -np.inf), 1)
            f[f"minx_{name}"] = np.nanmin(np.where(pres, p[..., 0], np.inf), 1)
        s = spd[:, m]
        f[f"spd_{name}"] = _nanmean(s, 1)
        with np.errstate(invalid="ignore"):
            f[f"slow_{name}"] = np.nansum(s < 1.0, 1) / np.maximum(np.sum(~np.isnan(s), 1), 1)
        cxv = np.full(T, np.nan, np.float32)
        cxv[2:-2] = (f[f"cx_{name}"][4:] - f[f"cx_{name}"][:-4]) / (4 * DT)
        f[f"cxv_{name}"] = cxv
    for k in ("maxx_a", "minx_a", "maxx_b", "minx_b"):
        f[k][~np.isfinite(f[k])] = np.nan
    f["spd_all"] = _nanmean(spd, 1)
    with np.errstate(invalid="ignore"):
        f["spd_med"] = np.nanmedian(spd, 1)
    f["dcx"] = f["cx_a"] - f["cx_b"]

    obs = ~np.isnan(ball[:, 0])
    f["ball_obs"] = obs.astype(np.float32)
    bspd = _central_speed(ball)
    f["ball_spd"] = bspd
    out = np.maximum(np.abs(ball[:, 0]) - L / 2, np.abs(ball[:, 1]) - W / 2)
    f["ball_out"] = out                                    # >0 outside the pitch
    f["ball_x"] = ball[:, 0]
    f["ball_ay"] = np.abs(ball[:, 1])
    dist = np.hypot(pl[..., 0] - ball[:, None, 0], pl[..., 1] - ball[:, None, 1])
    for k, name in ((0, "a"), (1, "b")):
        dk = np.where(present[:, team == k], dist[:, team == k], np.inf)
        ds = np.sort(dk, 1)
        n1 = ds[:, 0] if ds.shape[1] else np.full(T, np.inf)
        n2 = ds[:, 1] if ds.shape[1] > 1 else np.full(T, np.inf)
        n1[~np.isfinite(n1) | ~obs] = np.nan
        n2[~np.isfinite(n2) | ~obs] = np.nan
        f[f"d1_{name}"] = n1
        f[f"d2_{name}"] = n2
        with np.errstate(invalid="ignore"):
            f[f"c2_{name}"] = np.where(obs, (dk < 2).sum(1), np.nan).astype(np.float32)
            f[f"c5_{name}"] = np.where(obs, (dk < 5).sum(1), np.nan).astype(np.float32)
    f["margin"] = f["d1_b"] - f["d1_a"]                    # >0: team a nearer
    # control proxy: the nearest player moves with the ball
    near = np.argmin(np.where(present, dist, np.inf), 1)
    pv = np.full((T, 2), np.nan, np.float32)
    pv[2:-2] = (pl[4:, :, :] - pl[:-4, :, :])[np.arange(T - 4), near[2:-2]] / (4 * DT)
    bv = np.full((T, 2), np.nan, np.float32)
    bv[2:-2] = (ball[4:] - ball[:-4]) / (4 * DT)
    f["relv"] = np.hypot(*(pv - bv).T)
    f["near_is_a"] = np.where(obs, (team[near] == 0).astype(np.float32), np.nan)

    # v2, clear control: a player of one side within 1 m, nobody of the other within 2.5 m, slow
    # ball, ball seen. Its history says who had the ball before and who has it after a flight or a
    # gap, even when the (possibly mis-projected) flying ball sits near the wrong players.
    if VERSION >= 2:
        with np.errstate(invalid="ignore"):
            ok = obs & (np.nan_to_num(bspd, nan=99) < 6)
            ca = ok & (np.nan_to_num(f["d1_a"], nan=99) < 1.0) & (np.nan_to_num(f["d1_b"], nan=99) > 2.5)
            cb = ok & (np.nan_to_num(f["d1_b"], nan=99) < 1.0) & (np.nan_to_num(f["d1_a"], nan=99) > 2.5)
        t_ = np.arange(T)
        for name, m in (("a", ca), ("b", cb)):
            fi_, bi_ = _ffill_idx(m), _bfill_idx(m)
            f[f"sincectrl_{name}"] = np.where(fi_ >= 0, t_ - fi_, CTRL_CAP).clip(0, CTRL_CAP).astype(np.float32)
            f[f"untilctrl_{name}"] = np.where(bi_ < T, bi_ - t_, CTRL_CAP).clip(0, CTRL_CAP).astype(np.float32)
        f["ctrl_last_is_a"] = np.sign(f["sincectrl_b"] - f["sincectrl_a"]).astype(np.float32)
        f["ctrl_next_is_a"] = np.sign(f["untilctrl_b"] - f["untilctrl_a"]).astype(np.float32)
        f["ctrl_now"] = (ca.astype(np.float32) - cb.astype(np.float32))
        # where the ball is heading: nearest player of each side to the ball extrapolated 0.5 s / 1 s
        for hz, tag in ((12, "05"), (25, "10")):
            px = ball[:, 0] + bv[:, 0] * hz * DT
            py = ball[:, 1] + bv[:, 1] * hz * DT
            dd = np.hypot(pl[..., 0] - px[:, None], pl[..., 1] - py[:, None])
            for k, name in ((0, "a"), (1, "b")):
                dk = np.where(present[:, team == k], dd[:, team == k], np.inf).min(1) if (team == k).any() else np.full(T, np.inf)
                dk[~np.isfinite(dk)] = np.nan
                f[f"ahead{tag}_{name}"] = dk.astype(np.float32)
            f[f"ahead{tag}_m"] = f[f"ahead{tag}_b"] - f[f"ahead{tag}_a"]

    # v3, line of sight. A single camera puts a ball that is in the air on the ground further
    # along its viewing ray. The true ball (if a player could reach it, i.e. below header height)
    # lies on the ray's ground footprint between the observed point G and the point where the ray
    # is HEADER metres high. Distance of each side's nearest player to that segment; it equals the
    # usual nearest-player distance when the ray is short (camera high, ball near the camera foot).
    if VERSION >= 3:
        names3 = ("rayd_a", "rayd_b", "ray_m", "ray_len")
        if cam is None or not np.all(np.isfinite(cam)) or cam[2] <= HEADER:
            for k in names3:
                f[k] = np.full(T, np.nan, np.float32)
        else:
            smax = HEADER / cam[2]
            E = ball + smax * (cam[None, :2] - ball)
            seg = E - ball
            L2 = np.maximum((seg ** 2).sum(1), 1e-6)
            rel = pl - ball[:, None, :]
            tt = np.clip((rel * seg[:, None, :]).sum(2) / L2[:, None], 0, 1)
            near = ball[:, None, :] + tt[..., None] * seg[:, None, :]
            dd = np.hypot(*(pl - near).transpose(2, 0, 1))
            dd = np.where(present, dd, np.inf)
            for k, name in ((0, "a"), (1, "b")):
                m = dd[:, team == k].min(1) if (team == k).any() else np.full(T, np.inf)
                m[~np.isfinite(m) | ~obs] = np.nan
                f[f"rayd_{name}"] = m.astype(np.float32)
            f["ray_m"] = f["rayd_b"] - f["rayd_a"]
            f["ray_len"] = np.where(obs, np.sqrt(L2), np.nan).astype(np.float32)

    # gap context: where/how the ball was last seen and next seen, how long it is gone
    fi, bi = _ffill_idx(obs), _bfill_idx(obs)
    t = np.arange(T)
    since = np.where(fi >= 0, t - fi, MAXGAP).clip(0, MAXGAP)
    until = np.where(bi < T, bi - t, MAXGAP).clip(0, MAXGAP)
    f["since"] = since.astype(np.float32)
    f["until"] = until.astype(np.float32)
    f["gap"] = np.where(obs, 0, np.minimum(since + until, 2 * MAXGAP)).astype(np.float32)
    for tag, idx, ok in (("last", fi, fi >= 0), ("next", bi, bi < T)):
        j = np.clip(idx, 0, T - 1)
        for key in ("ball_out", "margin", "ball_spd", "ball_x", "ball_ay", "near_is_a"):
            f[f"{key}_{tag}"] = np.where(ok, f[key][j], np.nan).astype(np.float32)
    return f


def _pool(x):
    T = len(x) // STEP * STEP
    return _nanmean(x[:T].reshape(-1, STEP), 1)


def _roll(x, lo, hi):
    """NaN-aware mean of x over [t+lo, t+hi] (5 Hz steps)."""
    n = len(x)
    v = np.nan_to_num(x).astype(np.float64)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    i = np.arange(n)
    a = np.clip(i + lo, 0, n)
    b = np.clip(i + hi + 1, 0, n)
    cnt = cc[b] - cc[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[b] - cv[a]) / cnt, np.nan).astype(np.float32)


ROLL_KEYS = ("spd_all", "spd_med", "slow_a", "slow_b", "spd_a", "spd_b", "ball_obs", "ball_spd",
             "ball_out", "margin", "near_is_a", "relv", "cxv_a", "cxv_b", "dcx", "sx_a", "sx_b",
             "c2_a", "c2_b", "d1_a", "d1_b")


LONG_WINDOWS = {"c50": (-250, 250), "c150": (-750, 750)}          # +/-10 s, +/-30 s
LONG_KEYS = ("spd_all", "spd_med", "ball_obs", "ball_out", "near_is_a", "margin", "slow_a", "slow_b", "dcx")
V2_ROLL = ("ahead05_m", "ahead10_m", "ctrl_now")
V3_ROLL = ("ray_m",)


def build(d, swap=False):
    """(names, X (N,F) float32) at 5 Hz for one contiguous period."""
    f = frame_features(d, swap)
    pooled = {k: _pool(v.astype(np.float32)) for k, v in f.items()}
    cols = dict(pooled)
    for k in ROLL_KEYS + (V2_ROLL if VERSION >= 2 else ()) + (V3_ROLL if VERSION >= 3 else ()):
        for w, (lo, hi) in WINDOWS.items():
            cols[f"{k}_{w}"] = _roll(pooled[k], lo // STEP, hi // STEP)
    if VERSION >= 2:
        for k in LONG_KEYS:
            for w, (lo, hi) in LONG_WINDOWS.items():
                cols[f"{k}_{w}"] = _roll(pooled[k], lo // STEP, hi // STEP)
    names = sorted(cols)
    return names, np.column_stack([cols[k] for k in names]).astype(np.float32)


def pool_labels(lab):
    """5 Hz label = label of the block's centre frame."""
    T = len(lab) // STEP * STEP
    return lab[:T].reshape(-1, STEP)[:, STEP // 2]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, P = 500, 22
    d = {"pitch": (105.0, 68.0), "players_team": np.repeat([0, 1], 11).astype(np.int8),
         "players_xy": rng.normal(0, 15, (T, P, 2)).astype(np.float32),
         "ball_xy": rng.normal(0, 10, (T, 2)).astype(np.float32)}
    d["players_xy"][:, :11, 0] -= 10   # home deeper -> attacks +x
    d["ball_xy"][100:180] = np.nan
    n1, X1 = build(d)
    n2, X2 = build(d, swap=True)
    assert n1 == n2 and X1.shape == (T // STEP, len(n1))
    i, j = n1.index("n_a"), n1.index("n_b")
    assert np.allclose(X1[:, i], X2[:, j])               # swap exchanges teams
    g = n1.index("gap")
    assert X1[25, g] > 0 and X1[0, g] == 0
    # line of sight: a ball shown far beyond a player along the ray maps back onto that player
    d2 = {"pitch": (105.0, 68.0), "players_team": np.array([0, 1], np.int8), "cam": np.array([0.0, 64.0, 20.0]),
          "players_xy": np.tile(np.array([[[0.0, 0.0], [0.0, -10.0]]], np.float32), (50, 1, 1)),
          "ball_xy": np.tile(np.array([[0.0, -1.5]], np.float32), (50, 1))}   # ball 1.5 m below player a's feet on the ray
    VERSION = 3                                          # ray features exist from v3 on
    fr = frame_features(d2)
    assert fr["rayd_a"][10] < 0.01 and fr["rayd_b"][10] > 8, (fr["rayd_a"][10], fr["rayd_b"][10])
    print("features:", len(n1), "ok")
