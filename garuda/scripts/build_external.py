"""Rebuild external/processed/*.npz|json from the raw downloads in external/raw/. Deterministic.

    .venv/bin/python -I garuda/scripts/build_external.py [idsse|metrica|skillcorner ...]

Raw layout (download steps in the README, "Retrain"):
    external/raw/idsse/DFL_0{2_01,3_02,4_03}_*.xml       figshare 10.6084/m9.figshare.28196177
    external/raw/metrica/                                 git clone metrica-sports/sample-data
    external/raw/skillcorner/  + skillcorner_lfs/*.jsonl  git clone SkillCorner/opendata + LFS objects
Output format: see garuda/garuda/ext_data.py.
"""
import glob
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from lxml import etree

EXT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external"))
RAW = os.path.join(EXT, "raw")
OUT = os.path.join(EXT, "processed")
H, A, D = 0, 1, 2


def save(match_id, arrays, meta):
    os.makedirs(OUT, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    T = arrays["label"].size
    for k in ("frame", "period", "ball_xy", "ball_z", "players_xy"):
        assert arrays[k].shape[0] == T, k
    assert arrays["players_xy"].shape[1] == arrays["players_team"].size == arrays["player_ids"].size
    meta = dict(meta, match_id=match_id, n_frames=int(T), arrays=sorted(arrays))
    np.savez_compressed(os.path.join(OUT, match_id + ".npz"), **arrays)
    with open(os.path.join(OUT, match_id + ".json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(match_id, T, np.round(np.bincount(arrays["label"], minlength=3) / T, 3))


def gk_attack_dirs(players_xy, players_team, players_is_gk, period):
    """+1 if home attacks towards +x in that period, from the home GK's mean x (own goal side)."""
    out = {}
    for p in (1, 2):
        m = period == p
        gk = np.flatnonzero(players_is_gk & (players_team == 0))
        x = np.nanmean(players_xy[m][:, gk, 0]) if gk.size else np.nan
        out[str(p)] = None if np.isnan(x) else (1 if x < 0 else -1)
    return out


# IDSSE / DFL (Bassek et al. 2025)
def _iso_ms(s):
    return int(round(datetime.fromisoformat(s).timestamp() * 1000))


def build_idsse():
    for pos_path in sorted(glob.glob(os.path.join(RAW, "idsse", "DFL_04_03_positions_raw_observed_*.xml"))):
        mid = pos_path.rsplit("_", 1)[1][:-4]                       # DFL-MAT-J03WMX
        info_path = glob.glob(os.path.join(RAW, "idsse", "DFL_02_01_matchinformation_*%s.xml" % mid))[0]
        general = etree.parse(info_path).find(".//General")
        env = etree.parse(info_path).find(".//Environment")
        home_id, away_id = general.get("HomeTeamId"), general.get("GuestTeamId")
        gk_ids = {p.get("PersonId") for p in etree.parse(info_path).iter("Player")
                  if p.get("PlayingPosition") == "TW"}

        # framesets: (section, team, person) -> columns
        sets = {}
        pitch = None
        for _, el in etree.iterparse(pos_path, events=("end",), tag=("FrameSet", "PitchSize")):
            if el.tag == "PitchSize":
                pitch = (float(el.get("X")), float(el.get("Y")))
                continue
            key = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            fr = el.findall("Frame")
            cols = {"N": np.array([int(f.get("N")) for f in fr], np.int64),
                    "X": np.array([float(f.get("X")) for f in fr], np.float32),
                    "Y": np.array([float(f.get("Y")) for f in fr], np.float32)}
            if key[1].upper() == "BALL":
                cols["Z"] = np.array([float(f.get("Z", "nan")) for f in fr], np.float32)
                cols["BallStatus"] = np.array([int(f.get("BallStatus", -1)) for f in fr], np.int8)
                cols["BallPossession"] = np.array([int(f.get("BallPossession", 0)) for f in fr], np.int8)
                cols["T"] = np.array([_iso_ms(f.get("T")) for f in fr], np.int64)
                cols["attrs"] = sorted({a for f in fr for a in f.keys()})
            sets[key] = cols
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]

        sections = [s for s in ("firstHalf", "secondHalf") if any(k[0] == s for k in sets)]
        assert len(sections) == 2, sections
        ball = {s: next(v for k, v in sets.items() if k[0] == s and k[1].upper() == "BALL") for s in sections}
        frame = np.concatenate([ball[s]["N"] for s in sections])
        period = np.concatenate([np.full(ball[s]["N"].size, i + 1, np.int8) for i, s in enumerate(sections)])
        ball_xy = np.concatenate([np.c_[ball[s]["X"], ball[s]["Y"]] for s in sections])
        ball_z = np.concatenate([ball[s]["Z"] for s in sections])
        status = np.concatenate([ball[s]["BallStatus"] for s in sections])
        poss = np.concatenate([ball[s]["BallPossession"] for s in sections])
        time_ms = np.concatenate([ball[s]["T"] for s in sections])

        pids = sorted({(k[1], k[2]) for k in sets if k[1] in (home_id, away_id)},
                      key=lambda tp: (tp[0] != home_id, tp[1]))
        players_xy = np.full((frame.size, len(pids), 2), np.nan, np.float32)
        offset = 0
        players_outside_ball_frames = 0
        for i, s in enumerate(sections):
            n = ball[s]["N"]
            for j, (team, pid) in enumerate(pids):
                c = sets.get((s, team, pid))
                if c is None:
                    continue
                idx = np.searchsorted(n, c["N"])
                ok = (idx < n.size) & (n[np.minimum(idx, n.size - 1)] == c["N"])
                players_outside_ball_frames += int((~ok).sum())
                players_xy[offset + idx[ok], j] = np.c_[c["X"], c["Y"]][ok]
            offset += n.size
        players_team = np.array([0 if t == home_id else 1 for t, _ in pids], np.int8)
        players_is_gk = np.array([p in gk_ids for _, p in pids])

        # label: DEAD where BallStatus==0; otherwise BallPossession 1 -> home, 2 -> away
        # (code mapping verified against event Team attributes in scripts/report_external.py)
        label = np.where(status == 1, np.where(poss == 1, H, np.where(poss == 2, A, -1)), D).astype(np.int8)
        bad = int((label < 0).sum())
        assert bad == 0, "%d alive frames with BallPossession not in {1,2}" % bad

        save("idsse_" + mid.split("-")[-1], dict(
            frame=frame, period=period, ball_xy=ball_xy, ball_z=ball_z, players_xy=players_xy,
            players_team=players_team, players_is_gk=players_is_gk, player_ids=np.array([p for _, p in pids]),
            label=label, label_is_proxy=False, ball_status=status, ball_possession_raw=poss, time_ms=time_ms,
        ), dict(
            source="IDSSE/DFL open data (Bassek et al. 2025, Sci Data), figshare 10.6084/m9.figshare.28196177",
            source_files=[os.path.basename(p) for p in (pos_path, info_path)],
            fps=25, pitch_length=pitch[0], pitch_width=pitch[1],
            pitch_env=[float(env.get("PitchX")), float(env.get("PitchY"))],
            home_team=general.get("HomeTeamName"), away_team=general.get("GuestTeamName"),
            competition=general.get("CompetitionName"), season=general.get("Season"),
            matchday=general.get("MatchDay"), result=general.get("Result"),
            home_attacks_plus_x=gk_attack_dirs(players_xy, players_team, players_is_gk, period),
            coords="metres, origin pitch centre, x along length; native DFL coordinates, no transform",
            label_rule="D if BallStatus==0 else (H if BallPossession==1 else A)",
            ball_frame_attrs=ball[sections[0]]["attrs"],
            players_outside_ball_frames=players_outside_ball_frames,
            gk_source="matchinformation PlayingPosition=='TW'",
        ))


# Metrica sample data (proxy labels)
ON_BALL = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY"}   # events whose team = team in possession


def metrica_proxy_labels(frame, period, ev):
    """Event-derived proxy labels. ev: DataFrame(team in {0,1}, type, period, start, end) in frames.

    Possession: team of the last ON_BALL event started at or before the frame.
    Dead: from the end of the last non-CARD event before each SET PIECE to that SET PIECE's start
    (exclusive), plus everything before a period's first SET PIECE and after its last event ends.
    """
    label = np.full(frame.size, D, np.int8)
    for p in np.unique(period):
        m = np.flatnonzero(period == p)
        f = frame[m]
        e = ev[ev.period == p].sort_values(["start", "end"], kind="stable").reset_index(drop=True)
        on = e[e.type.isin(ON_BALL)]
        k = np.searchsorted(on.start.values, f, side="right") - 1
        lab = np.where(k >= 0, on.team.values[np.maximum(k, 0)], D).astype(np.int8)
        sp = e.index[e.type == "SET PIECE"]
        dead = np.zeros(f.size, bool)
        dead |= f < e.start[sp[0]]
        dead |= f > e[["start", "end"]].max(axis=1).max()
        for i in sp[1:]:
            prev = e.iloc[:i][e.type.iloc[:i] != "CARD"]
            t0 = max(prev.start.iloc[-1], prev.end.iloc[-1])
            dead |= (f >= t0) & (f < e.start[i])
        lab[dead] = D
        label[m] = lab
    return label


def _metrica_gk(players_xy, players_team, period):
    """Per team, the player whose mean |x| in period 1 is largest (keeper stands nearest a goal line)."""
    gk = np.zeros(players_team.size, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # players who never appear in period 1
        mx = np.abs(np.nanmean(players_xy[period == 1][:, :, 0], axis=0))
    for t in (0, 1):
        idx = np.flatnonzero((players_team == t) & ~np.isnan(mx))
        gk[idx[np.argmax(mx[idx])]] = True
    return gk


def build_metrica():
    L, W = 105.0, 68.0
    base = os.path.join(RAW, "metrica", "data")
    for g in (1, 2):
        d = os.path.join(base, "Sample_Game_%d" % g)
        tabs = []
        for side in ("Home", "Away"):
            path = os.path.join(d, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (g, side))
            hdr = pd.read_csv(path, skiprows=2, nrows=0).columns.tolist()
            names = []
            for c in hdr:                      # x column carries the name, the following one is y
                if c.startswith("Unnamed"):
                    names.append(names[-1][:-2] + "_y")
                else:
                    names.append(c + "_x" if c.startswith(("Player", "Ball")) else c)
            tabs.append(pd.read_csv(path, skiprows=3, header=None, names=names))
        home, away = tabs
        assert (home.Frame.values == away.Frame.values).all()
        frame = home.Frame.values.astype(np.int64)
        period = home.Period.values.astype(np.int8)
        cols = [(0, c[:-2]) for c in home.columns if c.startswith("Player") and c.endswith("_x")] + \
               [(1, c[:-2]) for c in away.columns if c.startswith("Player") and c.endswith("_x")]
        tab = {0: home, 1: away}
        players_xy = np.stack([np.c_[tab[t][n + "_x"].values, tab[t][n + "_y"].values] for t, n in cols], 1)
        players_xy = ((players_xy - 0.5) * [L, W]).astype(np.float32)
        ball_xy = ((np.c_[home.Ball_x.values, home.Ball_y.values] - 0.5) * [L, W]).astype(np.float32)
        players_team = np.array([t for t, _ in cols], np.int8)

        ev = pd.read_csv(os.path.join(d, "Sample_Game_%d_RawEventsData.csv" % g))
        ev = pd.DataFrame(dict(team=(ev.Team == "Away").astype(int), type=ev.Type, period=ev.Period,
                               start=ev["Start Frame"], end=ev["End Frame"]))
        label = metrica_proxy_labels(frame, period, ev)
        gk = _metrica_gk(players_xy, players_team, period)
        save("metrica_%d" % g, dict(
            frame=frame, period=period, ball_xy=ball_xy, ball_z=np.full(frame.size, np.nan, np.float32),
            players_xy=players_xy, players_team=players_team, players_is_gk=gk,
            player_ids=np.array(["%s_%s" % ("home" if t == 0 else "away", n) for t, n in cols]),
            label=label, label_is_proxy=True,
        ), dict(
            source="Metrica Sports sample-data, Sample_Game_%d (CSV format)" % g, fps=25,
            pitch_length=L, pitch_width=W, home_team="Home", away_team="Away",
            home_attacks_plus_x=gk_attack_dirs(players_xy, players_team, gk, period),
            coords="metres = (normalised - 0.5) * (105, 68); source origin top-left, y grows downward, "
                   "NOT flipped -- same convention as converting the competition ATD this way",
            label_rule="PROXY from events: see metrica_proxy_labels()",
            gk_source="heuristic: per team, largest mean |x| in period 1",
        ))

    # game 3: EPTS/FIFA tracking (same family as the competition ATD) + JSON events
    from kloppy import metrica
    d = os.path.join(base, "Sample_Game_3")
    ds = metrica.load_tracking_epts(meta_data=os.path.join(d, "Sample_Game_3_metadata.xml"),
                                    raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"))
    df = ds.to_df()
    frame = df.frame_id.values.astype(np.int64)
    period = df.period_id.values.astype(np.int8)
    teams = ds.metadata.teams
    plist = [(ti, p) for ti, t in enumerate(teams) for p in t.players if p.player_id + "_x" in df]
    players_xy = np.stack([df[[p.player_id + "_x", p.player_id + "_y"]].values.astype(float)
                           for _, p in plist], 1)
    players_xy = ((players_xy - 0.5) * [L, W]).astype(np.float32)
    ball_xy = ((df[["ball_x", "ball_y"]].values.astype(float) - 0.5) * [L, W]).astype(np.float32)
    players_team = np.array([ti for ti, _ in plist], np.int8)
    gk_meta = np.array([p.starting_position is not None and p.starting_position.code == "GK" for _, p in plist])
    gk_heur = _metrica_gk(players_xy, players_team, period)

    evj = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
    home_name = teams[0].name
    ev = pd.DataFrame(dict(team=[0 if e["team"]["name"] == home_name else 1 for e in evj],
                           type=[e["type"]["name"] for e in evj], period=[e["period"] for e in evj],
                           start=[e["start"]["frame"] for e in evj], end=[e["end"]["frame"] for e in evj]))
    label = metrica_proxy_labels(frame, period, ev)
    save("metrica_3", dict(
        frame=frame, period=period, ball_xy=ball_xy, ball_z=np.full(frame.size, np.nan, np.float32),
        players_xy=players_xy, players_team=players_team, players_is_gk=gk_meta,
        player_ids=np.array([p.player_id for _, p in plist]), label=label, label_is_proxy=True,
    ), dict(
        source="Metrica Sports sample-data, Sample_Game_3 (FIFA/EPTS tracking + JSON events)", fps=25,
        pitch_length=L, pitch_width=W, home_team=teams[0].name, away_team=teams[1].name,
        home_attacks_plus_x=gk_attack_dirs(players_xy, players_team, gk_meta, period),
        coords="metres = (normalised - 0.5) * (105, 68); source origin top-left, y grows downward, NOT flipped",
        label_rule="PROXY from events: see metrica_proxy_labels()",
        gk_source="EPTS metadata starting position GK; heuristic agrees: %s" % bool((gk_meta == gk_heur).all()),
    ))


# SkillCorner open data (10 fps)
def build_skillcorner():
    base = os.path.join(RAW, "skillcorner", "data", "matches")
    for mdir in sorted(glob.glob(os.path.join(base, "*"))):
        mid = os.path.basename(mdir)
        trk = os.path.join(RAW, "skillcorner_lfs", mid + "_tracking_extrapolated.jsonl")
        if not os.path.isdir(mdir) or not os.path.exists(trk):
            continue
        info = json.load(open(os.path.join(mdir, mid + "_match.json")))
        home_id = info["home_team"]["id"]
        team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in info["players"]}
        is_gk = {p["id"]: p["player_role"]["acronym"] == "GK" for p in info["players"]}
        recs = [json.loads(l) for l in open(trk)]
        recs = [r for r in recs if r["period"] in (1, 2)]
        frame = np.array([r["frame"] for r in recs], np.int64)
        period = np.array([r["period"] for r in recs], np.int8)
        nan = float("nan")
        bxyz = np.array([[r["ball_data"][k] if r["ball_data"][k] is not None else nan for k in "xyz"]
                         for r in recs], np.float32)
        pids = sorted({p["player_id"] for r in recs for p in r["player_data"]},
                      key=lambda i: (team_of[i], i))
        col = {p: j for j, p in enumerate(pids)}
        players_xy = np.full((frame.size, len(pids), 2), np.nan, np.float32)
        detected = np.zeros((frame.size, len(pids)), bool)
        for i, r in enumerate(recs):
            for p in r["player_data"]:
                players_xy[i, col[p["player_id"]]] = (p["x"], p["y"])
                detected[i, col[p["player_id"]]] = bool(p["is_detected"])
        group = np.array([{"home team": H, "away team": A}.get(r["possession"]["group"], -1) for r in recs], np.int8)
        ball_detected = np.array([bool(r["ball_data"]["is_detected"]) for r in recs])

        # alive := inside a phase-of-play interval (SkillCorner: phases exist only while ball in play)
        ph = pd.read_csv(os.path.join(mdir, mid + "_phases_of_play.csv"))
        alive = np.zeros(frame.size, bool)
        phase_team = np.full(frame.size, -1, np.int8)
        assert ph.team_in_possession_id.notna().all()
        for s, e, t in zip(ph.frame_start, ph.frame_end, ph.team_in_possession_id):
            m = (frame >= s) & (frame <= e)
            alive |= m
            phase_team[m] = H if t == home_id else A
        label = np.where(alive, phase_team, D).astype(np.int8)
        save("skillcorner_" + mid, dict(
            frame=frame, period=period, ball_xy=bxyz[:, :2], ball_z=bxyz[:, 2], players_xy=players_xy,
            players_team=np.array([team_of[p] for p in pids], np.int8),
            players_is_gk=np.array([is_gk[p] for p in pids]), player_ids=np.array([str(p) for p in pids]),
            label=label, label_is_proxy=True, possession_group=group, players_detected=detected,
            ball_detected=ball_detected,
        ), dict(
            source="SkillCorner opendata (A-League 2024/25 broadcast tracking), match %s" % mid, fps=10,
            pitch_length=float(info["pitch_length"]), pitch_width=float(info["pitch_width"]),
            home_team=info["home_team"]["name"], away_team=info["away_team"]["name"],
            home_attacks_plus_x={"1": 1 if info["home_team_side"][0] == "left_to_right" else -1,
                                 "2": 1 if info["home_team_side"][1] == "left_to_right" else -1},
            coords="metres, origin pitch centre, x along length (native SkillCorner)",
            label_rule="PROXY: D outside phases_of_play intervals; inside, team_in_possession of the phase. "
                       "Per-frame tracking possession.group kept separately as possession_group (-1 = null)",
            gk_source="match.json player_role GK",
        ))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["idsse", "metrica", "skillcorner"]
    for name in todo:
        globals()["build_" + name]()
