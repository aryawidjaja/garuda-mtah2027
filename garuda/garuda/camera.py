"""Camera centre from the per-frame pitch->image homography (Zhang 2000; research/AERIAL.md).

Assumptions: square pixels, zero skew, principal point at the image centre. Per frame:
  H' = T^-1 H S          S: centred metres -> the provider's normalised pitch, T: principal point
  focal from the two rotation constraints (orthogonal, equal-norm columns), solved jointly in
  least squares for u = 1/f^2 (one constraint degenerates when the camera faces the pitch square-on)
  [r1 r2 t] = lambda K^-1 H',  r3 = r1 x r2,  C = -R^T t
A fixed tactical camera pans and zooms but does not move, so the per-match estimate is the
median centre over valid frames. Only the centre is used downstream.
"""

import numpy as np


def _S(L, W):
    return np.array([[1 / L, 0, 0.5], [0, -1 / W, 0.5], [0, 0, 1]])


def centre_from_H(Hn, L, W, image=(1920, 1080)):
    """Camera centre (x, y, height) in centred metres for one normalised homography, or None."""
    T = np.array([[1, 0, image[0] / 2], [0, 1, image[1] / 2], [0, 0, 1.0]])
    Hp = np.linalg.solve(T, Hn @ _S(L, W))
    h1, h2, h3 = Hp[:, 0], Hp[:, 1], Hp[:, 2]
    a1, b1 = h1[0] * h2[0] + h1[1] * h2[1], h1[2] * h2[2]
    a2, b2 = h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2, h1[2] ** 2 - h2[2] ** 2
    den = a1 * a1 + a2 * a2
    if den <= 0:
        return None
    u = -(a1 * b1 + a2 * b2) / den
    if not np.isfinite(u) or u <= 0:
        return None
    f = 1 / np.sqrt(u)
    M = np.linalg.solve(np.diag([f, f, 1.0]), Hp)
    lam = 2 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:                                     # the pitch must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    C = -R.T @ t
    return np.array([C[0], C[1], abs(C[2])])


def match_camera(H_frames, L, W, max_frames=4000, seed=0):
    """Median camera centre over a random sample of valid frames, plus the share of frames that
    gave a usable estimate. H_frames: (N, 9) row-major normalised homographies (NaN = invalid)."""
    ok = np.flatnonzero(~np.isnan(H_frames[:, 0]))
    if not len(ok):
        return None, 0.0
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ok, size=min(max_frames, len(ok)), replace=False))
    cs = [c for c in (centre_from_H(H_frames[i].reshape(3, 3), L, W) for i in pick) if c is not None]
    if not cs:
        return None, 0.0
    cs = np.array(cs)
    good = (cs[:, 2] > 3) & (cs[:, 2] < 120) & (np.abs(cs[:, 0]) < 300) & (np.abs(cs[:, 1]) < 300)
    if good.sum() < 20:
        return None, float(good.mean())
    return np.median(cs[good], 0), float(good.mean())


def synthetic_H(C, look, f, L, W, image=(1920, 1080)):
    """Normalised homography of a pinhole camera at C (metres) looking at `look` (metres, z=0)."""
    C, look = np.asarray(C, float), np.asarray(look, float)
    z = look - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])                         # world -> camera
    t = -R @ C
    K = np.array([[f, 0, image[0] / 2], [0, f, image[1] / 2], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])  # metres -> pixels
    return Hm @ np.linalg.inv(_S(L, W))              # normalised -> pixels


if __name__ == "__main__":
    L, W = 105.3, 68.0
    rng = np.random.default_rng(0)
    true = np.array([0.0, -W / 2 - 30, 22.0])       # behind one touchline, 22 m up
    Hs = []
    for _ in range(300):                             # a panning, zooming camera
        look = [rng.uniform(-45, 45), rng.uniform(-20, 20), 0]
        Hn = synthetic_H(true, look, rng.uniform(1500, 4000), L, W)
        Hn = Hn / Hn[2, 2] * (1 + rng.normal(0, 1e-4, (3, 3)))
        Hs.append(Hn.ravel())
    c, share = match_camera(np.array(Hs), L, W)
    print("recovered", np.round(c, 2), "true", true, "usable %.2f" % share)
    assert np.allclose(c, true, atol=0.5), c
    print("ok")
