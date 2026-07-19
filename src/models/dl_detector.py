#!/usr/bin/env python3
"""
dl_detector.py — the deep multimodal anomaly detector.

An unsupervised reconstruction autoencoder trained ONLY on assumed-normal hours;
per-hour anomaly score = reconstruction MSE. Two fusion topologies:

    early — MLP autoencoder over the concatenated rich features.
    late  — per-modality encoders (audio / video / env) -> shared latent ->
            per-modality decoders. This is the fusion topology the alpha_t gate
            sits on; it is the natural home for the RICH feature set.

torch is imported lazily so the classical path runs without a GPU/torch install.

Interface shared with the classical detector:
    d = AEDetector(dims, kind="late").fit(X_normal, X_val)
    scores = d.score(X)          # higher = more anomalous (per row)
"""
from __future__ import annotations

import numpy as np


class AEDetector:
    def __init__(self, dims: dict[str, int], kind: str = "late", latent: int = 8,
                 epochs: int = 300, lr: float = 1e-3, patience: int = 30,
                 batch: int = 64, seed: int = 0, verbose: bool = False):
        self.dims = dims
        self.kind = kind
        self.name = f"DL-{kind}"
        self.latent = latent
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch = batch
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self._torch = None

    # -- lazy torch + model construction ------------------------------------
    def _build(self, d_total):
        import torch
        import torch.nn as nn
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)

        dims, latent = self.dims, self.latent

        class Early(nn.Module):
            def __init__(s):
                super().__init__()
                h = max(16, d_total)
                s.enc = nn.Sequential(nn.Linear(d_total, h), nn.ReLU(), nn.Linear(h, latent))
                s.dec = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, d_total))

            def forward(s, x):
                return s.dec(s.enc(x))

        class Late(nn.Module):
            def __init__(s):
                super().__init__()
                s.order = list(dims.keys())
                s.enc = nn.ModuleDict({m: nn.Sequential(
                    nn.Linear(dims[m], max(8, dims[m])), nn.ReLU(),
                    nn.Linear(max(8, dims[m]), latent)) for m in s.order})
                fused = latent * len(s.order)
                s.fuse = nn.Sequential(nn.Linear(fused, fused), nn.ReLU())
                s.dec = nn.ModuleDict({m: nn.Sequential(
                    nn.Linear(fused, max(8, dims[m])), nn.ReLU(),
                    nn.Linear(max(8, dims[m]), dims[m])) for m in s.order})

            def forward(s, x):
                parts, i = {}, 0
                for m in s.order:
                    parts[m] = x[:, i:i + dims[m]]; i += dims[m]
                z = s.fuse(s._torch_cat([s.enc[m](parts[m]) for m in s.order]))
                return s._torch_cat([s.dec[m](z) for m in s.order])

            def _torch_cat(s, xs):
                import torch as _t
                return _t.cat(xs, dim=1)

        return (Early() if self.kind == "early" else Late()).to(self.device)

    # -- fit / score --------------------------------------------------------
    def fit(self, X, Xval=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        X = np.asarray(X, dtype=np.float32)
        if Xval is None:
            k = max(1, int(0.15 * len(X)))
            Xval, X = X[:k], X[k:]
        Xval = np.asarray(Xval, dtype=np.float32)

        self.model = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=self.batch, shuffle=True)
        Xv = torch.tensor(Xval, device=self.device)

        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.model.train()
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = loss_fn(self.model(Xv), Xv).item()
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in self.model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 25 == 0:
                print(f"    [{self.name}] epoch {ep:3d} val_mse={v:.4f}")
        if best_state:
            self.model.load_state_dict(best_state)
        if self.verbose:
            print(f"    [{self.name}] best val_mse={best:.4f}")
        return self

    def score(self, X):
        import torch
        self.model.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            err = ((self.model(Xt) - Xt) ** 2).mean(dim=1)
        return err.cpu().numpy()
