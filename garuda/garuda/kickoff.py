"""Nothing before a half's kickoff is live play.

The scored second-half block of BAR-ATH opens before its kickoff (FAQ.md: those frames are all D),
and warm-up balls on the pitch can make a tracker report a "ball" near players. Each half's
kickoff is found without frame constants and every scored frame before it is labelled D:
  1. a Smart Tagging SET PIECES window of type "Kick Off (Start)" opening near the start of the
     scored half block (official notebook: these windows contain the real kickoff);
  2. inside it, the first frame where the ball is within NEAR m of the centre spot and has moved
     more than MOVE m one second later;
  3. otherwise the window's start, which is never after the true kickoff, so the fallback can
     only under-apply the rule.
No such tag: nothing is applied.
"""

import numpy as np

NEAR, MOVE, LOOK = 3.0, 2.0, 25          # metres, metres, frames (1 s)
SEARCH_BEFORE, SEARCH_AFTER = 30 * 25, 6 * 60 * 25   # tag may open shortly before / within 6 min of block start


def kickoff_frame(tags, ball_xy, first, last):
    """Video frame of the kickoff for the scored block [first, last], or None."""
    cand = [t for t in tags if t["code"] == "set pieces" and t["type"] == "kick off (start)"
            and first - SEARCH_BEFORE <= t["start"] <= first + SEARCH_AFTER]
    if not cand:
        return None
    t = min(cand, key=lambda x: x["start"])
    a, b = max(t["start"], 0), min(t["end"] + LOOK, len(ball_xy) - LOOK - 1)
    for f in range(a, b):
        p, q = ball_xy[f], ball_xy[f + LOOK]
        if np.isfinite(p).all() and np.hypot(*p) < NEAR and np.isfinite(q).all() and np.hypot(*(q - p)) > MOVE:
            # the ball rests near the spot at f and is kicked within 1 s: the kickoff is the first
            # frame it has left its resting point (> 0.3 m), not f itself
            for g in range(f, f + LOOK + 1):
                if np.isfinite(ball_xy[g]).all() and np.hypot(*(ball_xy[g] - p)) > 0.3:
                    return int(g)
            return int(f)
    return int(t["start"])


def apply(labels, tags, ball_xy, first, last, dead="D"):
    """In place on a full-length label array: D for scored frames before the detected kickoff."""
    k = kickoff_frame(tags, ball_xy, first, last)
    if k is not None and k > first:
        labels[first:min(k, last + 1)] = dead
    return k


if __name__ == "__main__":
    T = 4000
    ball = np.full((T, 2), np.nan, np.float32)
    ball[:1500] = [[20.0, 5.0]]                  # warm-up ball elsewhere
    ball[1500:2000] = [[0.0, 0.0]]               # placed on the centre spot
    ball[2000:] = np.column_stack([0.4 * np.arange(1, T - 2000 + 1), np.zeros(T - 2000)])   # kicked at 2000, 10 m/s
    tags = [{"code": "set pieces", "type": "kick off (start)", "start": 1910, "end": 2040, "team": "", "half": ""}]
    lab = np.array(["H"] * T, dtype="<U1")
    k = apply(lab, tags, ball, 1000, 3999)
    assert k == 2000, k                          # the first frame the ball has left the spot
    assert (lab[1000:k] == "D").all() and (lab[k:] == "H").all() and (lab[:1000] == "H").all()
    lab2 = np.array(["H"] * T, dtype="<U1")
    assert apply(lab2, [], ball, 1000, 3999) is None and (lab2 == "H").all()   # no tag -> no change
    ball3 = np.full((T, 2), np.nan, np.float32)  # ball never seen -> fall back to the tag start
    lab3 = np.array(["H"] * T, dtype="<U1")
    assert apply(lab3, tags, ball3, 1000, 3999) == 1910 and (lab3[1910:] == "H").all()
    print("ok")
