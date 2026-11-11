"""Smart Tagging -> additive emission bonuses for the decoder (5 Hz).

Only restart-type evidence is used. Tactical phases (BUILD UP, ...) are left out: their team tag
names the attacking side of a phase, not the frame-level owner, and teams that tried them report
little or no gain.

Inside a SET PIECES window of length n, at relative position u in [0, 1]:
    bonus[D]          += beta  * (1 - u)       stoppage evidence fades as the window proceeds
    bonus[team]       += gamma * u             the tagged side takes the restart (if known)
After a GOALS window: D until the next kick-off window starts (cap 90 s).

simulate() draws Metrica-like windows from known restarts of other matches so beta/gamma can be
chosen without touching the competition match: windows open ~3.5 s before the restart (published
for both kick-off tags in the official notebook), only part of restarts are tagged, and some
carry no team (every throw-in does in BAR-ATH).
"""

import numpy as np

STEP = 5


def bonus_from_windows(windows, n5, beta, gamma, sign_home):
    """windows: [(start25, end25, team) team in {0, 1, None}] already in this period's frames.
    sign_home maps a tagged team to the decoder's state index (0 = H, 1 = A)."""
    b = np.zeros((n5, 3), np.float32)
    for s, e, tm in windows:
        s5, e5 = max(0, s // STEP), min(n5, max(s // STEP + 1, e // STEP))
        if e5 <= s5:
            continue
        u = np.linspace(0, 1, e5 - s5, dtype=np.float32)
        b[s5:e5, 2] += beta * (1 - u)
        if tm is not None:
            b[s5:e5, tm] += gamma * u
    return b


def goal_bonus(goal_ends, kickoff_starts, n5, beta):
    b = np.zeros((n5, 3), np.float32)
    for g in goal_ends:
        nxt = [k for k in kickoff_starts if k > g]
        stop = min(nxt[0] if nxt else g + 90 * 25, g + 90 * 25)
        b[g // STEP:min(n5, stop // STEP), 2] += beta
    return b


def restarts(label):
    """(frame, team) of every dead -> alive change in a 25 Hz label array."""
    i = np.flatnonzero((label[:-1] == 2) & (label[1:] != 2)) + 1
    return [(int(f), int(label[f])) for f in i]


def simulate(label, rng, recall=0.5, lead_s=(2.5, 4.5), len_s=(5.0, 10.0), na_p=0.35):
    out = []
    for f, tm in restarts(label):
        if rng.random() > recall:
            continue
        s = f - int(rng.uniform(*lead_s) * 25)
        e = s + int(rng.uniform(*len_s) * 25)
        out.append((max(0, s), e, None if rng.random() < na_p else tm))
    return out


def windows_from_tags(tags, start_frame, stop_frame, home_name, away_name):
    """Competition tags -> (set-piece windows, goal ends, kick-off starts) in period-local frames."""
    def team(t):
        if t in (home_name, ):
            return 0
        if t in (away_name, ):
            return 1
        return None
    sp, goals, ko = [], [], []
    for t in tags:
        if not (start_frame <= t["start"] < stop_frame):
            continue
        s, e = t["start"] - start_frame, t["end"] - start_frame
        if t["code"] == "set pieces":
            sp.append((s, e, team(t["team"])))
            if t["type"].startswith("kick off"):
                ko.append(s)
        elif t["code"] == "goals":
            goals.append(e)
    return sp, goals, sorted(ko)


if __name__ == "__main__":
    lab = np.array([2] * 100 + [0] * 300 + [2] * 50 + [1] * 200)
    assert restarts(lab) == [(100, 0), (450, 1)]
    w = simulate(lab, np.random.default_rng(0), recall=1.0, na_p=0.0)
    assert len(w) == 2 and w[0][2] == 0
    b = bonus_from_windows(w, len(lab) // STEP, 2.0, 1.0, None)
    assert b[:, 2].max() > 1.5 and b[:, 0].max() > 0
    print("ok")
