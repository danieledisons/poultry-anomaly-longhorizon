#!/usr/bin/env python3
"""
dl_gate_pytorch.py — REAL DL version of the gate head-to-head (run on the GPU box).

Same story, same evaluation as dl_gate_headtohead_proxy.py, but the anomaly
scorer is a trained PyTorch autoencoder instead of PCA. Everything else — the
detrended lean fusion features, the alpha_t persistence gate, the synthetic
injection protocol, the output format — is identical, so the narrative figures
regenerate with real numbers.

Models (unsupervised reconstruction; trained ONLY on assumed-normal fusion hours):
    early   — MLP autoencoder over the concatenated lean features.
    late    — per-modality encoders (audio / video / env) -> shared latent ->
              per-modality decoders. Reconstruction error is the fused residual.
              This is the fusion topology the alpha_t gate sits on top of.

Anomaly score per hour = reconstruction MSE. The SAME gate then runs on that
error stream and only closes under SUSTAINED unexplained energy.

Claims tested:
  1. DETREND (day_index growth removed) > RAW               -> injection ROC-AUC
  2. DL + alpha_t gate > DL alone (event-level P/R/F1)       -> the core thesis plot

Outputs (drop them back into chat):
    results/dl_pytorch_auc.csv          per-row injection AUC, RAW vs DETREND x model
    results/dl_pytorch_dl_vs_gate.csv   event-level P/R/F1, DL alone vs DL+gate
    results/dl_pytorch_dl_vs_gate.png   the head-to-head bar chart (600 dpi)
    results/dl_pytorch_auc.png          AUC-vs-magnitude curves (600 dpi)
    models are checkpointed to results/*.pt (gitignored)

Run
---
    python src/pipeline/dl_gate_pytorch.py --model late --variant DETREND \
        --epochs 300 --trials 300
    # or sweep everything:
    python src/pipeline/dl_gate_pytorch.py --model both --variant both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf   # only for the RAW-vs-DETREND AUC baseline
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import AlphaGate, robust_scale
from src.pipeline.fusion_compare import robust_standardize, detrend_dayindex

GATE_PARAMS = dict(deadband=1.0, decay=0.85, close_threshold=6.0,
                   open_threshold=2.0, per_step_cap=2.0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Models
# ======================================================================
class EarlyAE(nn.Module):
    """MLP autoencoder over concatenated features."""
    def __init__(self, d, latent=8):
        super().__init__()
        h = max(16, d)
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(dim=1)


class LateAE(nn.Module):
    """Per-modality encoders -> shared latent -> per-modality decoders.
    This is the late-fusion topology the gate sits on top of."""
    def __init__(self, dims: dict[str, int], latent=8):
        super().__init__()
        self.dims = dims
        self.order = list(dims.keys())
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(dims[m], max(8, dims[m])), nn.ReLU(),
                             nn.Linear(max(8, dims[m]), latent))
            for m in self.order})
        fused = latent * len(self.order)
        self.fuse = nn.Sequential(nn.Linear(fused, fused), nn.ReLU())
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(fused, max(8, dims[m])), nn.ReLU(),
                             nn.Linear(max(8, dims[m]), dims[m]))
            for m in self.order})

    def _split(self, x):
        out, i = {}, 0
        for m in self.order:
            out[m] = x[:, i:i + self.dims[m]]; i += self.dims[m]
        return out

    def forward(self, x):
        parts = self._split(x)
        z = self.fuse(torch.cat([self.encoders[m](parts[m]) for m in self.order], dim=1))
        return torch.cat([self.decoders[m](z) for m in self.order], dim=1)

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(dim=1)


# ======================================================================
# Training
# ======================================================================
def train_ae(model, Xtr, Xval, epochs=300, lr=1e-3, patience=30, batch=64, verbose=True):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32)),
                    batch_size=batch, shuffle=True)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=DEVICE)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for (xb,) in tr:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xval_t), Xval_t).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"  epoch {ep:3d}  val_mse={v:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  best val_mse={best:.4f}")
    return model


def build_model(kind, model_feats, dims):
    if kind == "early":
        return EarlyAE(len(model_feats))
    return LateAE(dims)


# ======================================================================
# Data prep
# ======================================================================
def prep(spine, features_csv, variant):
    m = pd.read_csv(spine, parse_dates=["time"])
    feats = pd.read_csv(features_csv)["feature"].tolist()
    model_feats = [f for f in feats if f != "env_day_index"]
    fus = (m[m["coverage_state"] == "both_lit"]
           .sort_values("time").dropna(subset=model_feats).reset_index(drop=True))
    X = fus[model_feats].to_numpy(float)
    if variant == "DETREND":
        X = detrend_dayindex(X, fus["env_day_index"].to_numpy(float))
    X = robust_standardize(X)
    dims = {"aud": sum(f.startswith("aud_") for f in model_feats),
            "vid": sum(f.startswith("vid_") for f in model_feats),
            "env": sum(f.startswith("env_") for f in model_feats)}
    dims = {k: v for k, v in dims.items() if v > 0}
    # reorder columns to match dims grouping (aud, vid, env) for LateAE splitting
    order = [f for f in model_feats if f.startswith("aud_")] \
          + [f for f in model_feats if f.startswith("vid_")] \
          + [f for f in model_feats if f.startswith("env_")]
    idx = [model_feats.index(f) for f in order]
    return X[:, idx], order, dims


# ======================================================================
# Evaluation (identical protocol to the proxy)
# ======================================================================
def injection_auc(scorer, Xtr, Xte, mags, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mm in mags:
        signs = rng.choice([-1.0, 1.0], size=Xte.shape)
        Xp = Xte + mm * signs
        y = np.r_[np.zeros(len(Xte)), np.ones(len(Xp))]
        s = np.r_[scorer(Xte), scorer(Xp)]
        rows.append((mm, roc_auc_score(y, s)))
    return pd.DataFrame(rows, columns=["magnitude", "auc"])


def dl_vs_gate(scorer, X, clean_err, trials=300, seed=0,
               n_sustained=3, sustained_len=(12, 30), sustained_mag=2.5,
               n_spikes=5, spike_mag=10.0, timeline=240):
    rng = np.random.default_rng(seed)
    thr = np.quantile(clean_err, 0.99)
    scale = robust_scale(np.sqrt(clean_err) - np.median(np.sqrt(clean_err)))
    n = len(X)
    c = dict(tp_dl=0, fp_dl=0, fn_dl=0, tp_g=0, fp_g=0, fn_g=0)
    for _ in range(trials):
        T = X[rng.integers(0, n, size=timeline)].copy()
        slots = rng.permutation(np.arange(20, timeline - 40, 8))
        spans, spikes = [], []
        for s in slots[:n_sustained]:
            L = int(rng.integers(*sustained_len))
            T[s:s + L] += sustained_mag * rng.choice([-1., 1.], size=(1, X.shape[1]))
            spans.append((s, s + L))
        for s in slots[n_sustained:n_sustained + n_spikes]:
            T[s] += spike_mag * rng.choice([-1., 1.], size=X.shape[1]); spikes.append(s)
        err = scorer(T)
        dl_flag = err > thr
        resid = np.sqrt(err) - np.median(np.sqrt(err))
        closed = AlphaGate(scale=scale, **GATE_PARAMS).run(resid)["closed"]
        for a, b in spans:
            c["tp_dl" if dl_flag[a:b].any() else "fn_dl"] += 1
            c["tp_g" if closed[a:b].any() else "fn_g"] += 1
        for p in spikes:
            if dl_flag[p:p + 2].any(): c["fp_dl"] += 1
            if closed[p:p + 2].any(): c["fp_g"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.
        r = tp / (tp + fn) if tp + fn else 0.
        return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else (0, 0, 0)
    dl, g = prf(c["tp_dl"], c["fp_dl"], c["fn_dl"]), prf(c["tp_g"], c["fp_g"], c["fn_g"])
    return pd.DataFrame([
        {"method": "DL alone",  "precision": dl[0], "recall": dl[1], "f1": dl[2],
         "sustained_TP": c["tp_dl"], "spike_FP": c["fp_dl"], "sustained_FN": c["fn_dl"]},
        {"method": "DL + gate", "precision": g[0],  "recall": g[1],  "f1": g[2],
         "sustained_TP": c["tp_g"],  "spike_FP": c["fp_g"],  "sustained_FN": c["fn_g"]},
    ])


# ======================================================================
def run_one(kind, variant, args):
    print(f"\n########## model={kind}  variant={variant}  device={DEVICE} ##########")
    X, order, dims = prep(args.spine, args.features, variant)
    torch.manual_seed(0); np.random.seed(0)
    n = len(X); cut = int(n * 0.7); vcut = int(n * 0.85)
    Xtr, Xval, Xte = X[:cut], X[cut:vcut], X[vcut:]
    print(f"features={X.shape[1]}  dims={dims}  train/val/test={len(Xtr)}/{len(Xval)}/{len(Xte)}")

    model = build_model(kind, order, dims)
    model = train_ae(model, Xtr, Xval, epochs=args.epochs)

    def scorer(A):
        model.eval()
        return model.recon_error(torch.tensor(A, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    torch.save(model.state_dict(), str(RESULTS_DIR / f"ae_{kind}_{variant}.pt"))
    clean_err = scorer(np.r_[Xtr, Xval])

    auc = injection_auc(scorer, Xtr, Xte, [1, 2, 3, 4, 6]); auc.insert(0, "model", kind); auc.insert(0, "variant", variant)
    hh = dl_vs_gate(scorer, X, clean_err, trials=args.trials); hh.insert(0, "model", kind); hh.insert(0, "variant", variant)
    return auc, hh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine", default=str(RESULTS_DIR / "spine_room2_rich.csv"))
    ap.add_argument("--features", default=str(RESULTS_DIR / "recommended_features.csv"))
    ap.add_argument("--model", default="both", choices=["early", "late", "both"])
    ap.add_argument("--variant", default="both", choices=["RAW", "DETREND", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["early", "late"] if args.model == "both" else [args.model]
    variants = ["RAW", "DETREND"] if args.variant == "both" else [args.variant]

    aucs, hhs = [], []
    for v in variants:
        for k in kinds:
            a, h = run_one(k, v, args); aucs.append(a); hhs.append(h)
    auc = pd.concat(aucs, ignore_index=True); hh = pd.concat(hhs, ignore_index=True)
    auc.to_csv(RESULTS_DIR / "dl_pytorch_auc.csv", index=False)
    hh.to_csv(RESULTS_DIR / "dl_pytorch_dl_vs_gate.csv", index=False)

    print("\n=== INJECTION ROC-AUC (RAW vs DETREND) ===")
    print(auc.pivot_table(index=["model", "magnitude"], columns="variant", values="auc").round(3).to_string())
    print("\n=== DL alone vs DL + alpha_t gate (event-level) ===")
    print(hh.to_string(index=False))

    # figures (600 dpi)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # head-to-head bars (use best variant = DETREND if present)
        vsel = "DETREND" if "DETREND" in variants else variants[0]
        sub = hh[hh.variant == vsel]
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), squeeze=False)
        for ax, k in zip(axes[0], kinds):
            d = sub[sub.model == k]; mets = ["precision", "recall", "f1"]; x = np.arange(3); w = .35
            ax.bar(x - w/2, d[d.method == "DL alone"][mets].values[0], w, label="DL alone", color="#9e9e9e")
            ax.bar(x + w/2, d[d.method == "DL + gate"][mets].values[0], w, label="DL + gate", color="#2e7d32")
            ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in mets]); ax.set_ylim(0, 1.05)
            ax.set_title(f"{k} fusion ({vsel})", fontweight="bold"); ax.grid(axis="y", alpha=.3); ax.legend()
        fig.suptitle("DL alone vs DL + alpha_t gate (PyTorch)", fontweight="bold"); fig.tight_layout()
        fig.savefig(RESULTS_DIR / "dl_pytorch_dl_vs_gate.png", dpi=600); plt.close(fig)

        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), squeeze=False, sharey=True)
        for ax, k in zip(axes[0], kinds):
            for v in variants:
                d = auc[(auc.model == k) & (auc.variant == v)]
                ax.plot(d.magnitude, d.auc, "o-", label=v)
            ax.set_title(f"{k}"); ax.set_xlabel("injection magnitude (sigma)"); ax.set_ylim(.45, 1.02)
            ax.grid(alpha=.3); ax.axhline(.5, color="gray", lw=.6, ls="--"); ax.legend(title="features")
        axes[0][0].set_ylabel("ROC-AUC")
        fig.suptitle("Injection detection: RAW vs DETREND (PyTorch)", fontweight="bold"); fig.tight_layout()
        fig.savefig(RESULTS_DIR / "dl_pytorch_auc.png", dpi=600); plt.close(fig)
        print("\nWrote dl_pytorch_auc.csv/.png and dl_pytorch_dl_vs_gate.csv/.png")
    except ImportError:
        print("\n(matplotlib missing — CSVs written, figures skipped)")


if __name__ == "__main__":
    main()
