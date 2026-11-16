"""Small bidirectional GRU over the per-row feature table (5 Hz), blended with the LightGBM probabilities (v4).
It reads a whole half as one sequence instead of one row and its fixed windows."""
import numpy as np
import torch
from torch import nn

CH, BATCH, EPOCHS = 256, 32, 8


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n, 64), nn.ReLU())
        self.gru = nn.GRU(64, 48, batch_first=True, bidirectional=True)
        self.out = nn.Linear(96, 3)

    def forward(self, x):
        return self.out(self.gru(self.inp(x))[0])


def train(Xs, ys, ws):
    """Weighted cross-entropy on 256-row chunks, seed 0. Returns (net, mu, sd, prior)."""
    X = np.concatenate(Xs)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    del X
    cx, cy, cw = [], [], []
    for x, y, w in zip(Xs, ys, ws):
        n = len(x) // CH * CH
        cx.append(np.nan_to_num((x[:n] - mu) / sd).astype(np.float32).reshape(-1, CH, x.shape[1]))
        cy.append(y[:n].reshape(-1, CH)); cw.append(np.full(n // CH, w, np.float32))
    cx, cy, cw = (torch.from_numpy(np.concatenate(a)) for a in (cx, cy, cw))
    cy = cy.long()
    torch.manual_seed(0)
    net = Net(cx.shape[2])
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    g = torch.Generator().manual_seed(0)
    for ep in range(EPOCHS):
        tot = 0.0
        for idx in torch.randperm(len(cx), generator=g).split(BATCH):
            l = loss_fn(net(cx[idx]).reshape(-1, 3), cy[idx].reshape(-1)).reshape(len(idx), CH).mean(1)
            l = (l * cw[idx]).sum() / cw[idx].sum()
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l) * len(idx)
        print(f"  epoch {ep} loss {tot / len(cx):.4f}", flush=True)
    yy = np.concatenate([np.full(len(y), w) for y, w in zip(ys, ws)])
    prior = np.bincount(np.concatenate(ys), weights=yy, minlength=3) / yy.sum()
    return net.eval(), mu, sd, prior


def predict(model, X, per):
    """Class probabilities, one GRU pass per period (per = period id of each row)."""
    net, mu, sd, _ = model
    out = np.zeros((len(X), 3), np.float32)
    with torch.no_grad():
        for k in np.unique(per):
            m = per == k
            x = torch.from_numpy(np.nan_to_num((X[m] - mu) / sd).astype(np.float32))[None]
            out[m] = torch.softmax(net(x)[0], -1).numpy()
    return out


def save(model, path):
    net, mu, sd, prior = model
    torch.save({"state": net.state_dict(), "mu": torch.from_numpy(mu),
                "sd": torch.from_numpy(sd), "prior": torch.from_numpy(prior)}, path)


def load(path):
    d = torch.load(path, weights_only=True)
    net = Net(len(d["mu"])); net.load_state_dict(d["state"])
    return net.eval(), d["mu"].numpy(), d["sd"].numpy(), d["prior"].numpy()
