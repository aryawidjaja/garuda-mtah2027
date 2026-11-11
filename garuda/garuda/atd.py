"""Readers for the competition files (BAR-ATH). Run on the participant's machine only.

Output uses the garuda.ext_data dict layout, so one feature pipeline serves both:
    fps, pitch (L, W), ball_xy (T,2) metres centred (NaN = not tracked), players_xy (T,P,2),
    players_team (P,) 0 home / 1 away, frame (T,) submission frame index.

ATD format (official starter/baseline.py), one line per video frame:
    "<frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>"  players then ball, 0-1 normalised,
    NaN where missing. frame_id counts from 1; submission frame = frame_id - 1.
Team of each slot from the metadata XML: the first <Team> is home.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

PITCH = (105.3, 68.0)  # games.py, measured from corrected tracking
FPS = 25


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in %s" % meta_path)
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    np.int8)


def _xy(group):
    if group[:1] in ("N", ""):
        return np.nan, np.nan
    a = group.split(",")
    return float(a[0]), float(a[1])


def read_atd(meta_path, raw_path, n_frames, pitch=PITCH, cache=None):
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files} | {"fps": FPS, "pitch": pitch}
    teams = slot_teams(meta_path)
    P = len(teams)
    L, W = pitch
    pl = np.full((n_frames, P, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            for s, g in enumerate(parts[1].split(";")[:P]):
                x, y = _xy(g)
                pl[f, s, 0] = (x - 0.5) * L
                pl[f, s, 1] = (0.5 - y) * W
            x, y = _xy(parts[2])
            ball[f] = ((x - 0.5) * L, (0.5 - y) * W)
    out = {"ball_xy": ball, "players_xy": pl, "players_team": teams,
           "frame": np.arange(n_frames, dtype=np.int32)}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **out)
    return out | {"fps": FPS, "pitch": pitch}


def read_homography_centre(path, n_frames, pitch=PITCH, image=(1920, 1080)):
    """Per-frame camera aim and zoom: the pitch point (metres, centred) under the image centre and
    the visible width (m) along the image's horizontal midline. NaN where invalid."""
    import csv
    L, W = pitch
    aim = np.full((n_frames, 2), np.nan, np.float32)
    span = np.full(n_frames, np.nan, np.float32)
    pts = np.array([[image[0] / 2, image[1] / 2, 1.0], [0.0, image[1] / 2, 1.0],
                    [image[0], image[1] / 2, 1.0]])
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            f = int(row[0])
            if not 0 <= f < n_frames or row[-1].strip() != "1":
                continue
            try:
                Hm = np.array([float(v) for v in row[1:10]]).reshape(3, 3)
                q = np.linalg.solve(Hm, pts.T).T
            except (ValueError, np.linalg.LinAlgError):
                continue
            q = q[:, :2] / q[:, 2:3]
            m = np.column_stack(((q[:, 0] - 0.5) * L, (0.5 - q[:, 1]) * W))
            aim[f] = m[0]
            span[f] = np.hypot(*(m[2] - m[1]))
    return aim, span


def read_homographies(path, n_frames):
    """(n, 9) row-major normalised pitch->pixel homographies; NaN rows where invalid or absent."""
    import csv
    Hm = np.full((n_frames, 9), np.nan, np.float32)
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            f = int(row[0])
            if 0 <= f < n_frames and row[-1].strip() == "1":
                Hm[f] = [float(v) for v in row[1:10]]
    return Hm


def read_valid(path, n_frames):
    """(n,) homography `valid` flag (1/0), NaN for rows absent from the file."""
    import csv
    v = np.full(n_frames, np.nan, np.float32)
    with open(path) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        j = hdr.index("valid")
        for row in rd:
            f = int(row[0])
            if 0 <= f < n_frames:
                v[f] = float(row[j] == "1")
    return v


def read_tags(path, fps=FPS):
    """[{code, start, end (frames, half-open), team, type, half}] from the SportsCode XML,
    case-normalised (codes/values lower-case)."""
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        tags = {(lb.findtext("group") or "").strip().lower(): (lb.findtext("text") or "").strip().lower()
                for lb in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": tags.get("team", ""), "type": tags.get("type", ""), "half": tags.get("half", ""),
        })
    return out


if __name__ == "__main__":
    # self-check on a synthetic two-frame file (never on match data)
    import tempfile
    d = tempfile.mkdtemp()
    meta = os.path.join(d, "m.xml")
    raw = os.path.join(d, "r.txt")
    open(meta, "w").write('<M><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>'
                          '<Player teamId="T1"/><Player teamId="T2"/></Players></M>')
    open(raw, "w").write("1:0.5,0.5,1.0;NaN,NaN,NaN:0.6,0.25,3.0\n2:0.0,1.0,0.0;0.5,0.5,0:NaN,NaN,NaN\n")
    r = read_atd(meta, raw, 2)
    assert list(r["players_team"]) == [0, 1]
    assert np.allclose(r["players_xy"][0, 0], [0, 0]) and np.isnan(r["players_xy"][0, 1]).all()
    assert np.allclose(r["ball_xy"][0], [0.1 * 105.3, 0.25 * 68.0], atol=1e-4)
    assert np.allclose(r["players_xy"][1, 0], [-52.65, -34.0]) and np.isnan(r["ball_xy"][1]).all()
    print("ok")
