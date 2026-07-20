#!/usr/bin/env python3
"""Compares a few DL/classical models on the merged Room 2 features.

Run: python src/pipeline/dl_model_comparison.py
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from src.models.alpha_gate import AlphaGate

try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

VIDEO_COL, AUDIO_COL, TEMP_COL, TEMP_TREND_COL = \
    "vid_flow_mean_avg", "aud_voc_frac_mean", "env_temp_day_mean_c", "env_temp_roll_mean_c"
DAY_START, DAY_END = 6, 18
WIN = 24            # window length (hours) for temporal models
EPOCHS = 120       # max training epochs (early-stopping usually halts sooner)
PATIENCE = 12      # early-stopping patience on validation loss
SEED = 7


# ---------------------------------------------------------------- data
def load_multimodal(merged_csv):
    d = pd.read_csv(merged_csv); d["time"] = pd.to_datetime(d["time"]); d["hh"] = d["time"].dt.hour
    f = (d[(d["hh"] >= DAY_START) & (d["hh"] < DAY_END)]
         .dropna(subset=[VIDEO_COL, AUDIO_COL, TEMP_COL]).sort_values("time").reset_index(drop=True))

    def z(x):
        x = x.values.astype(float); m = np.median(x); s = np.median(np.abs(x - m)) * 1.4826
        return (x - m) / max(s, 1e-9)
    Z = np.c_[z(f[VIDEO_COL]), z(f[AUDIO_COL]), z(f[TEMP_COL] - f[TEMP_TREND_COL])]
    # covariance of the correlated cluster (V,A) for the Mahalanobis models
    VA = Z[:, :2]; d0 = np.sqrt((VA ** 2).sum(1)); core = VA[d0 < np.quantile(d0, 0.90)]
    Sig = np.cov(core.T); Sinv = np.linalg.inv(Sig)
    return Z, Sig, Sinv


def calibrate(clean):
    q95 = np.quantile(clean, 0.95); med = np.median(clean)
    mad = np.median(np.abs(clean - med)) * 1.4826; s = max(mad, 1e-9)
    return dict(scale=s, deadband=q95 / s, decay=0.85, close_threshold=6.0,
               open_threshold=2.0, per_step_cap=2.0)


# ---------------------------------------------------------------- GRU autoencoder
if HAVE_TORCH:
    class RNNAutoencoder(nn.Module):
        """Seq2seq reconstruction autoencoder; rnn_type in {'gru','lstm'}."""
        def __init__(self, n_feat=3, hidden=16, latent=8, rnn_type="gru"):
            super().__init__()
            self.rnn_type = rnn_type
            RNN = nn.GRU if rnn_type == "gru" else nn.LSTM
            self.enc = RNN(n_feat, hidden, batch_first=True)
            self.to_lat = nn.Linear(hidden, latent)
            self.from_lat = nn.Linear(latent, hidden)
            self.dec = RNN(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_feat)

        def forward(self, x):                       # x: (B, T, F)
            if self.rnn_type == "gru":
                _, h = self.enc(x); henc = h[-1]
            else:
                _, (h, _) = self.enc(x); henc = h[-1]
            z = self.to_lat(henc)                   # (B, L)
            h0 = torch.tanh(self.from_lat(z)).unsqueeze(0)  # (1,B,H)
            rep = h0.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # (B,T,H)
            if self.rnn_type == "gru":
                dec, _ = self.dec(rep, h0)
            else:
                c0 = torch.zeros_like(h0); dec, _ = self.dec(rep, (h0, c0))
            return self.out(dec)

    def make_windows(Z, win=WIN):
        return np.stack([Z[i:i+win] for i in range(len(Z) - win + 1)])

    def train_rnn(Z, rnn_type="gru", win=WIN, epochs=EPOCHS, patience=PATIENCE, seed=SEED):
        torch.manual_seed(seed)
        X = torch.tensor(make_windows(Z, win), dtype=torch.float32)
        n = len(X); idx = np.arange(n)
        rng = np.random.default_rng(seed); rng.shuffle(idx)
        n_val = max(8, int(0.15 * n)); val_i, tr_i = idx[:n_val], idx[n_val:]
        model = RNNAutoencoder(n_feat=Z.shape[1], rnn_type=rnn_type)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
        lossf = nn.MSELoss()
        best, best_state, wait, best_ep = np.inf, None, 0, 0
        tag = f"{rnn_type.upper()}-AE"
        print(f"[{tag}] training (max {epochs} epochs, early-stop patience {patience}) ...")
        for ep in range(epochs):
            model.train(); tr = tr_i.copy(); np.random.shuffle(tr)
            tl = 0.0; nb = 0
            for b in range(0, len(tr), 64):
                xb = X[tr[b:b+64]]
                opt.zero_grad(); loss = lossf(model(xb), xb); loss.backward(); opt.step()
                tl += loss.item(); nb += 1
            model.eval()
            with torch.no_grad():
                vloss = float(lossf(model(X[val_i]), X[val_i]))
            improved = vloss < best - 1e-5
            if improved:
                best, best_state, wait, best_ep = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
            else:
                wait += 1
            # progress line every epoch (carriage-return updated, marks best + patience)
            print(f"\r[{tag}] epoch {ep+1:3d}/{epochs}  train {tl/max(nb,1):.4f}  "
                  f"val {vloss:.4f}  best {best:.4f}@{best_ep+1}  patience {wait}/{patience}"
                  f"{'  *' if improved else '   '}", end="", flush=True)
            if wait >= patience:
                print(f"\n[{tag}] early-stopped: no val improvement for {patience} epochs.")
                break
        else:
            print()  # newline after the last epoch if we didn't early-stop
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        print(f"[{tag}] done — best epoch {best_ep+1}, val MSE {best:.4f}")
        return model

    def rnn_window_error(model, window):            # window: (T,F) -> scalar error
        with torch.no_grad():
            x = torch.tensor(window[None], dtype=torch.float32)
            return float(((model(x) - x) ** 2).mean())

    def rnn_series_errors(model, W, win=WIN):
        """Per-timestep reconstruction error for a whole series, BATCHED
        (all sliding windows in one forward pass) — much faster than per-window."""
        n = len(W)
        if n < win:
            return np.full(n, np.nan)
        idx = np.arange(n - win + 1)
        batch = np.stack([W[i:i+win] for i in idx])          # (Nw, win, F)
        with torch.no_grad():
            x = torch.tensor(batch, dtype=torch.float32)
            rec = model(x)
            we = ((rec - x) ** 2).mean(dim=(1, 2)).cpu().numpy()  # (Nw,)
        errs = np.full(n, np.nan)
        errs[idx + win // 2] = we
        return errs


# ---------------------------------------------------------------- MLP autoencoder (sklearn)
class MLPAutoencoder:
    def __init__(self, seed=SEED):
        from sklearn.neural_network import MLPRegressor
        self.net = MLPRegressor(hidden_layer_sizes=(8, 4, 8), activation="tanh",
                                max_iter=800, random_state=seed)

    def fit(self, Z):
        self.net.fit(Z, Z); return self

    def point_error(self, Z):
        rec = self.net.predict(Z)
        return ((rec - Z) ** 2).mean(axis=1)


# ---------------------------------------------------------------- helpers
def maha(Z2, Sinv):
    return np.sqrt(np.einsum("ij,jk,ik->i", Z2, Sinv, Z2))


def cusum(series, k=0.5):
    """One-sided CUSUM on a nonneg deviation series; returns running statistic."""
    s = np.zeros(len(series)); acc = 0.0
    for t, v in enumerate(series):
        acc = max(0.0, acc + (v - k)); s[t] = acc
    return s


def roc_auc(scores, labels):
    order = np.argsort(-scores); y = labels[order]
    P = labels.sum(); N = len(labels) - P
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp / max(P, 1)]; fpr = np.r_[0, fp / max(N, 1)]
    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------- evaluation
def evaluate(Z, Sig, Sinv, out_dir):
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(Sig)
    gp_joint = calibrate(maha(Z[:, :2][np.sqrt((Z[:, :2] ** 2).sum(1)) <
                             np.quantile(np.sqrt((Z[:, :2] ** 2).sum(1)), .9)], Sinv))

    # fit learned models on the real (nominal) data
    iforest = None
    try:
        from sklearn.ensemble import IsolationForest
        iforest = IsolationForest(random_state=SEED, contamination=0.1).fit(Z)
    except Exception as e:
        print("[warn] IsolationForest unavailable:", e)
    mlp = MLPAutoencoder().fit(Z)
    gru = train_rnn(Z, "gru") if HAVE_TORCH else None
    lstm = train_rnn(Z, "lstm") if HAVE_TORCH else None
    if gru is None:
        print("[note] torch not present -> GRU/LSTM-AE rows skipped (run on your machine with torch).")

    def synth_window(kind, n=120, t0=48, dur=36):
        # FAIR base: a real contiguous multimodal window (same realistic distribution
        # for every model, incl. the autoencoders trained on real data). Inject on top.
        start = rng.integers(0, len(Z) - n)
        W = Z[start:start+n].copy()
        if kind == "sustained":
            c = rng.uniform(0.8, 2.0); W[t0:t0+dur, 0] += c; W[t0:t0+dur, 1] += c
        elif kind == "spike":
            W[t0, 0] += 12; W[t0, 1] += 12          # brief, large, 1h
        return W, slice(t0, t0+dur)

    # score functions: window -> continuous anomaly score, + optional gate-based flag/latency
    def score_maha_gate(W, w):
        d = maha(W[:, :2], Sinv); g = AlphaGate(**gp_joint).run(d)
        lat = int(np.argmax(g["closed"][w])) if g["closed"][w].any() else np.nan
        return g["persistence"][w].max(), g["closed"][w].any(), lat
    def score_maha_point(W, w):
        d = maha(W[:, :2], Sinv); return d[w].max(), (d > np.quantile(maha(Z[:, :2], Sinv), .999)).any(), np.nan
    def score_iforest(W, w):
        if iforest is None: return np.nan, False, np.nan
        sc = -iforest.decision_function(W)          # higher = more anomalous
        thr = np.quantile(-iforest.decision_function(Z), .95)
        return sc[w].max(), (sc[w] > thr).any(), np.nan
    def score_cusum(W, w):
        d = maha(W[:, :2], Sinv); c = cusum(d, k=np.median(maha(Z[:, :2], Sinv)) + 1)
        thr = 6.0
        cw = c[w]; return cw.max(), (cw > thr).any(), (int(np.argmax(cw > thr)) if (cw > thr).any() else np.nan)
    def score_mlp_point(W, w):
        e = mlp.point_error(W); thr = np.quantile(mlp.point_error(Z), .95)
        return e[w].max(), (e[w] > thr).any(), np.nan
    def _rnn_errs(net, W):
        errs = rnn_series_errors(net, W)              # batched forward pass
        return pd.Series(errs).interpolate().bfill().ffill().values
    def score_rnn_gate(net):
        def f(W, w):
            if net is None: return np.nan, False, np.nan
            errs = _rnn_errs(net, W)
            gp = calibrate(errs[:48][np.isfinite(errs[:48])])
            g = AlphaGate(**gp).run(errs)
            lat = int(np.argmax(g["closed"][w])) if g["closed"][w].any() else np.nan
            return g["persistence"][w].max(), g["closed"][w].any(), lat
        return f
    def score_rnn_point(net):
        def f(W, w):
            if net is None: return np.nan, False, np.nan
            errs = _rnn_errs(net, W); thr = np.quantile(errs, .95)
            return errs[w].max(), (errs[w] > thr).any(), np.nan
        return f

    models = {
        "GRU-AE + alpha_t (ours)": score_rnn_gate(gru),
        "GRU-AE (error only)": score_rnn_point(gru),
        "LSTM-AE + alpha_t (ours)": score_rnn_gate(lstm),
        "LSTM-AE (error only)": score_rnn_point(lstm),
        "MLP-AE (error only)": score_mlp_point,
        "Mahalanobis + alpha_t (ours)": score_maha_gate,
        "Mahalanobis (pointwise)": score_maha_point,
        "Isolation Forest": score_iforest,
        "CUSUM": score_cusum,
    }

    NP, NN = 300, 300
    rows = []
    for name, fn in models.items():
        # Battery A: sustained detection (AUC + latency)
        pos = [fn(*synth_window("sustained")) for _ in range(NP)]
        neg = [fn(*synth_window("clean")) for _ in range(NN)]
        sp = np.array([p[0] for p in pos]); sn = np.array([q[0] for q in neg])
        if np.all(np.isnan(sp)):
            rows.append(dict(model=name, auc_sustained=np.nan, recall=np.nan,
                             latency_h=np.nan, spike_falsealarm=np.nan)); continue
        scores = np.r_[sp, sn]; labels = np.r_[np.ones(NP), np.zeros(NN)]
        good = np.isfinite(scores)
        auc = roc_auc(scores[good], labels[good])
        recall = np.mean([p[1] for p in pos])
        lat = np.nanmean([p[2] for p in pos if not np.isnan(p[2])]) if any(not np.isnan(p[2]) for p in pos) else np.nan
        # Battery B: brief-spike false alarm
        spikes = [fn(*synth_window("spike")) for _ in range(NP)]
        fa = np.mean([s[1] for s in spikes])
        rows.append(dict(model=name, auc_sustained=round(auc, 3),
                         recall=round(float(recall), 3),
                         latency_h=(round(float(lat), 1) if np.isfinite(lat) else np.nan),
                         spike_falsealarm=round(float(fa), 3)))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print("\n=== MODEL COMPARISON (same injection harness) ===")
    print(tbl.to_string(index=False))
    return tbl


def make_figure(tbl, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = tbl.dropna(subset=["auc_sustained"]).copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(t))
    ax[0].barh(y, t["auc_sustained"], color="#1f4e79")
    ax[0].set_yticks(y); ax[0].set_yticklabels(t["model"], fontsize=8)
    ax[0].set_xlabel("ROC-AUC (sustained coupling-breaks)"); ax[0].set_xlim(0.4, 1.02)
    ax[0].set_title("Detection of subtle SUSTAINED breaks\n(higher better)", fontsize=10, fontweight="bold")
    ax[0].invert_yaxis(); ax[0].grid(alpha=.3, axis="x")
    ax[1].barh(y, t["spike_falsealarm"], color="#c62828")
    ax[1].set_yticks(y); ax[1].set_yticklabels(t["model"], fontsize=8)
    ax[1].set_xlabel("false-alarm rate on BRIEF spikes"); ax[1].set_xlim(0, 1.02)
    ax[1].set_title("False alarms on brief 12σ spikes\n(lower better — persistence gate wins)", fontsize=10, fontweight="bold")
    ax[1].invert_yaxis(); ax[1].grid(alpha=.3, axis="x")
    fig.suptitle("αₜ persistence-gating vs baselines", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_model_comparison.png"), dpi=600, bbox_inches="tight")
    plt.close(fig); print("[fig] fig_model_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(RESULTS_DIR / "room2_merged_hourly.csv"))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    print(f"[env] torch available: {HAVE_TORCH}")
    Z, Sig, Sinv = load_multimodal(args.merged)
    tbl = evaluate(Z, Sig, Sinv, args.out_dir)
    make_figure(tbl, args.out_dir)
    print("[done]", args.out_dir)


if __name__ == "__main__":
    main()