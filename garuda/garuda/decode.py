"""H/A/D sequence decoding with minimum durations, posterior sampling and spellF1
minimum-Bayes-risk (MBR) selection.

Hidden semi-Markov model as an expanded-state HMM: label L is a chain of m_L states walked in
order (minimum duration m_L steps), ending in a tail state with self-loop p_L (geometric
remainder). Leaving the tail enters label L' with probability q[L, L'].
Hybrid emissions (Bourlard & Morgan): alpha * (log p(L | x_t) - log prior(L)) from a per-frame
classifier; alpha < 1 tempers evidence that is autocorrelated between neighbouring frames.

  viterbi()        single best path
  sample_paths()   forward-filtering backward-sampling, n posterior paths
  mbr_select()     candidate with the highest mean spellF1 against the sampled paths
                   (estimated expected score)
"""

import numpy as np

H, A, D = 0, 1, 2
NL = 3


class HSMM:
    def __init__(self, min_dur, stay, switch, init=None):
        """min_dur: (K,) ints >= 1. stay: (K,) tail self-loop probs. switch: (K,K) rows = from-label
        distribution over next label (diagonal ignored, renormalised; zeros forbid a move).
        K = 3 (H, A, D) or 4 (H, A, D-home-restarts, D-away-restarts)."""
        self.min_dur = np.asarray(min_dur, int)
        NL = len(self.min_dur)
        if init is None:
            init = np.full(NL, 0.05)
            init[2:] = 0.9 / (NL - 2)
        stay = np.asarray(stay, float)
        q = np.array(switch, float)
        np.fill_diagonal(q, 0.0)
        q = q / q.sum(1, keepdims=True)
        self.NL = NL
        self.offsets = np.concatenate(([0], np.cumsum(self.min_dur)))[:NL]
        self.n_states = int(self.min_dur.sum())
        self.label_of = np.repeat(np.arange(NL), self.min_dur)
        src, dst, lp = [], [], []
        for L in range(NL):
            o, m = self.offsets[L], self.min_dur[L]
            for k in range(m - 1):
                src.append(o + k); dst.append(o + k + 1); lp.append(0.0)
            tail = o + m - 1
            src.append(tail); dst.append(tail); lp.append(np.log(stay[L]))
            for L2 in range(NL):
                if L2 != L and q[L, L2] > 0:
                    src.append(tail); dst.append(self.offsets[L2])
                    lp.append(np.log(1 - stay[L]) + np.log(q[L, L2]))
        order = np.lexsort((np.array(src), np.array(dst)))
        self.src = np.array(src)[order]
        self.dst = np.array(dst)[order]
        self.lp = np.array(lp)[order]
        # reduceat boundaries per destination state (every state has >= 1 predecessor)
        self.starts = np.searchsorted(self.dst, np.arange(self.n_states))
        # edges where possession passes straight between the teams (H <-> A); a per-frame bonus can
        # be added to them (hand-over evidence makes switching cheap only where it is likely)
        ls, ld = self.label_of[self.src], self.label_of[self.dst]
        self.team_switch = (ls < 2) & (ld < 2) & (ls != ld)
        init = np.asarray(init, float)
        self.log_init = np.full(self.n_states, -np.inf)
        for L in range(NL):
            self.log_init[self.offsets[L]] = np.log(init[L] / init.sum())

    def _step(self, prev, reduce_max, bonus=0.0):
        cand = prev[self.src] + self.lp
        if bonus:
            cand = cand + bonus * self.team_switch
        mx = np.maximum.reduceat(cand, self.starts)
        if reduce_max:
            return mx, cand
        safe = np.where(np.isfinite(mx), mx, 0.0)
        s = np.add.reduceat(np.exp(cand - safe[self.dst]), self.starts)
        with np.errstate(divide="ignore"):
            return safe + np.log(s), cand

    def forward(self, emis, sw=None):
        """emis: (T, K) emission log-scores; sw: optional (T,) team-switch log-bonus.
        Returns log alpha (T, S)."""
        T = len(emis)
        la = np.empty((T, self.n_states))
        la[0] = self.log_init + emis[0, self.label_of]
        for t in range(1, T):
            la[t] = self._step(la[t - 1], False, 0.0 if sw is None else sw[t])[0] + emis[t, self.label_of]
        return la

    def viterbi(self, emis, sw=None):
        T = len(emis)
        back = np.empty((T, self.n_states), np.int32)
        v = self.log_init + emis[0, self.label_of]
        for t in range(1, T):
            mx, cand = self._step(v, True, 0.0 if sw is None else sw[t])
            # argmax within each destination group
            is_max = cand == mx[self.dst]
            first = np.full(self.n_states, -1)
            idx = np.flatnonzero(is_max)
            first[self.dst[idx[::-1]]] = self.src[idx[::-1]]
            back[t] = first
            v = mx + emis[t, self.label_of]
        path = np.empty(T, np.int32)
        path[-1] = int(np.argmax(v))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return self.label_of[path]

    def sample_paths(self, emis, n, seed=0, la=None, sw=None):
        """Forward-filtering backward-sampling. Returns (n, T) label paths."""
        rng = np.random.default_rng(seed)
        if la is None:
            la = self.forward(emis, sw)
        T = len(emis)
        # predecessor lists per destination, for the backward pass
        preds = [(self.src[self.dst == s], self.lp[self.dst == s], self.team_switch[self.dst == s])
                 for s in range(self.n_states)]
        out = np.empty((n, T), np.int8)
        last = la[-1] - np.logaddexp.reduce(la[-1])
        p_last = np.exp(last)
        states = rng.choice(self.n_states, size=n, p=p_last / p_last.sum())
        out[:, -1] = self.label_of[states]  # state labels; map with to_hda() for 4-state models
        u = rng.random((T, n))
        for t in range(T - 1, 0, -1):
            for i in range(n):
                srcs, lps, tsw = preds[states[i]]
                w = la[t - 1, srcs] + lps + (0.0 if sw is None else sw[t] * tsw)
                w = np.exp(w - w.max())
                c = np.cumsum(w)
                states[i] = srcs[min(np.searchsorted(c, u[t, i] * c[-1]), len(srcs) - 1)]
            out[:, t - 1] = self.label_of[states]
        return out


def emissions(proba, prior, alpha):
    """(T,3) log emission scores from classifier posteriors."""
    p = np.clip(proba, 1e-6, 1.0)
    return alpha * (np.log(p) - np.log(np.asarray(prior)))


def emissions4(proba, prior, alpha, p_restart_home, beta=1.0):
    """(T,4): H, A, D-home-restarts, D-away-restarts. p_restart_home = P(home restarts | x, dead)."""
    e = emissions(proba, prior, alpha)
    r = np.clip(p_restart_home, 1e-4, 1 - 1e-4)
    return np.column_stack([e[:, 0], e[:, 1], e[:, 2] + beta * np.log(r),
                            e[:, 2] + beta * np.log(1 - r)])


SWITCH4 = np.array([[0, 1, 1, 1],      # H -> A, or a stoppage restarted by either side
                    [1, 0, 1, 1],
                    [1, 0, 0, 0.2],    # D-home-restarts ends in H (or relabels its restarter)
                    [0, 1, 0.2, 0]], float)


def to_hda(states):
    """Map 4-state labels to H/A/D."""
    return np.minimum(np.asarray(states), 2)


# spellF1 on label arrays, same definition as the official possession_metrics

def spells(labels):
    """[(start, stop, code)] maximal H or A runs of a contiguous label array."""
    labels = np.asarray(labels)
    if not len(labels):
        return []
    b = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], b))
    stops = np.concatenate((b, [len(labels)]))
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, stops) if labels[s] != D]


IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def spell_counts(gt_sp, pr_sp):
    """(tp per IoU threshold (5,), n_gt, n_pred). Greedy one-to-one by IoU desc, like the scorer."""
    ov = []
    j0 = 0
    for i, (gs, ge, gc) in enumerate(gt_sp):
        while j0 < len(pr_sp) and pr_sp[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pr_sp) and pr_sp[j][0] < ge:
            ps, pe, pc = pr_sp[j]
            if pc == gc:
                ov.append(((min(ge, pe) - max(gs, ps)) / (max(ge, pe) - min(gs, ps)), i, j))
            j += 1
    ov.sort(key=lambda x: (-x[0], x[1], x[2]))
    tps = np.zeros(len(IOUS), int)
    for k, th in enumerate(IOUS):
        g, p = set(), set()
        for iou, i, j in ov:
            if iou < th:
                break
            if i in g or j in p:
                continue
            g.add(i); p.add(j)
        tps[k] = len(g)
    return tps, len(gt_sp), len(pr_sp)


def spell_f1_from_counts(tps, n_gt, n_pr):
    tps = np.asarray(tps, float)
    denom = n_gt + n_pr
    return float(np.mean(np.where(tps > 0, 2 * tps / max(denom, 1), 0.0)))


def spell_f1(gt, pred):
    return spell_f1_from_counts(*spell_counts(spells(gt), spells(pred)))


def mbr_select(candidates, samples):
    """Index of the candidate with the highest mean spellF1 against the sampled paths, and the
    expected scores. Rewards whole-possession agreement, which a per-frame posterior cannot."""
    sp_samples = [spells(s) for s in samples]
    scores = []
    for c in candidates:
        sc = spells(c)
        scores.append(np.mean([spell_f1_from_counts(*spell_counts(s, sc)) for s in sp_samples]))
    return int(np.argmax(scores)), np.array(scores)


if __name__ == "__main__":
    # self-check: a noisy synthetic sequence; decoding must beat raw argmax on spellF1, and the
    # spellF1 implementation must agree with the official scorer on a random pair.
    import os, sys
    rng = np.random.default_rng(1)
    segs = [(D, 40), (H, 300), (A, 120), (H, 60), (D, 80), (A, 400), (H, 200), (D, 50), (H, 150)]
    truth = np.concatenate([np.full(n, L) for L, n in segs])
    T = len(truth)
    proba = np.full((T, 3), 0.15)
    proba[np.arange(T), truth] = 0.7
    flip = rng.random(T) < 0.35
    proba[flip] = rng.dirichlet(np.ones(3), flip.sum())
    raw = proba.argmax(1)
    m = HSMM(min_dur=(10, 10, 10), stay=(0.99, 0.99, 0.98),
             switch=[[0, .6, .4], [.6, 0, .4], [.5, .5, 0]])
    e = emissions(proba, (1 / 3, 1 / 3, 1 / 3), 1.0)
    vit = m.viterbi(e)
    smp = m.sample_paths(e, 16, seed=0)
    best, sc = mbr_select([vit] + list(smp), smp)
    print("raw %.3f  viterbi %.3f  mbr %.3f" % (spell_f1(truth, raw), spell_f1(truth, vit),
                                                  spell_f1(truth, ([vit] + list(smp))[best])))
    assert spell_f1(truth, vit) > spell_f1(truth, raw)
    assert spell_f1(truth, truth) == 1.0
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "official-repo", "evaluation"))
    try:
        import possession_metrics as pm
        r = pm.spell_scores(truth.astype(np.int64), raw.astype(np.int64), [(0, T)])["mean_f1"]
        assert abs(r - spell_f1(truth, raw)) < 1e-12, (r, spell_f1(truth, raw))
        r = pm.spell_scores(truth.astype(np.int64), vit.astype(np.int64), [(0, T)])["mean_f1"]
        assert abs(r - spell_f1(truth, vit)) < 1e-12
        print("matches official spell_scores")
    except ImportError:
        print("official scorer not found; skipped parity check")
    print("ok")
