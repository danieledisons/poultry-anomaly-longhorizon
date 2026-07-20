#!/usr/bin/env python3
"""Statistical proof for the cross-modal coherence claim: is the top canonical
correlation between the audio and video residual vectors real, or an artefact? Reports,
per room, the coherence r with a bootstrap CI, a permutation-null p-value (shuffle the
time alignment -> coupling should vanish), and the lag profile. Emits a proof figure.

Run: python src/pipeline/coherence_proof.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

OUT = RESULTS_DIR / "dev_conditioned" / "coherence"
MODEL = RESULTS_DIR / "dev_conditioned" / "model"


def load(room):
    df = pd.read_csv(MODEL / f"residuals_room{room}.csv", parse_dates=["time"])
    ac = [c for c in df.columns if c.startswith("resid_aud_")]
    vc = [c for c in df.columns if c.startswith("resid_vid_")]
    A = df[ac].to_numpy(float); V = df[vc].to_numpy(float)
    k = ~(np.isnan(A).any(1) | np.isnan(V).any(1))
    t = df["time"].to_numpy()[k]
    return A[k], V[k], t


def canon_r(A, V, k=5):
    Ap = PCA(k).fit_transform((A - A.mean(0)) / (A.std(0) + 1e-9))
    Vp = PCA(k).fit_transform((V - V.mean(0)) / (V.std(0) + 1e-9))
    U, W = CCA(1).fit(Ap, Vp).transform(Ap, Vp)
    return abs(np.corrcoef(U[:, 0], W[:, 0])[0, 1]), U[:, 0], W[:, 0]


def lag_profile(u, w, maxlag=8):
    out = []
    for L in range(-maxlag, maxlag + 1):
        if L < 0:   c = np.corrcoef(u[:L], w[-L:])[0, 1]
        elif L > 0: c = np.corrcoef(u[L:], w[:-L])[0, 1]
        else:       c = np.corrcoef(u, w)[0, 1]
        out.append((L, c))
    return out


def analyse(room, n_boot=500, n_perm=500, seed=0):
    A, V, t = load(room)
    rng = np.random.default_rng(seed)
    r, u, w = canon_r(A, V)
    n = len(A)
    boots = np.array([canon_r(A[i := rng.integers(0, n, n)], V[i])[0] for _ in range(n_boot)])
    perms = np.array([canon_r(A, V[rng.permutation(n)])[0] for _ in range(n_perm)])
    p = (1 + (perms >= r).sum()) / (n_perm + 1)
    return dict(room=room, n=n,
                t0=str(pd.Timestamp(t.min()).date()), t1=str(pd.Timestamp(t.max()).date()),
                r=r, ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                null_med=float(np.median(perms)), null_p95=float(np.percentile(perms, 95)),
                p_value=p, u=u, w=w, boots=boots, perms=perms, lags=lag_profile(u, w))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = [analyse(r) for r in (2, 6)]
    tbl = pd.DataFrame([{k: v for k, v in d.items()
                         if k not in ("u", "w", "boots", "perms", "lags")} for d in res])
    tbl.to_csv(OUT / "coherence_proof.csv", index=False)
    print(tbl.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    col = {2: "#2e7d32", 6: "#c46a2f"}

    # (a) observed r vs permutation null
    for d in res:
        ax[0, 0].hist(d["perms"], bins=30, alpha=0.45, color=col[d["room"]],
                      label=f"Room {d['room']} null")
        ax[0, 0].axvline(d["r"], color=col[d["room"]], lw=2.4,
                         label=f"Room {d['room']} observed r={d['r']:.2f} (p={d['p_value']:.3f})")
    ax[0, 0].set_title("(a) Observed coherence vs time-shuffled null")
    ax[0, 0].set_xlabel("top canonical correlation"); ax[0, 0].set_ylabel("count")
    ax[0, 0].legend(fontsize=7, frameon=False)

    # (b) bootstrap distribution of r
    for d in res:
        ax[0, 1].hist(d["boots"], bins=30, alpha=0.5, color=col[d["room"]],
                      label=f"Room {d['room']}: {d['r']:.2f} [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}]")
    ax[0, 1].set_title("(b) Bootstrap 95% CI of coherence")
    ax[0, 1].set_xlabel("top canonical correlation"); ax[0, 1].set_ylabel("count")
    ax[0, 1].legend(fontsize=7, frameon=False)

    # (c) lag profile
    for d in res:
        L, C = zip(*d["lags"])
        ax[1, 0].plot(L, C, "-o", ms=3, color=col[d["room"]], label=f"Room {d['room']}")
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_title("(c) Coherence vs audio->video lag")
    ax[1, 0].set_xlabel("lag (hours, +ve = audio leads video)")
    ax[1, 0].set_ylabel("canonical correlation"); ax[1, 0].legend(fontsize=8, frameon=False)

    # (d) energy-scalar vs canonical-variate scatter (Room 2) — shows where coherence lives
    d = res[0]; A, V, _ = load(2)
    a = np.linalg.norm(A, axis=1); v = np.linalg.norm(V, axis=1)
    ax[1, 1].scatter(np.argsort(np.argsort(d["u"])), np.argsort(np.argsort(d["w"])),
                     s=8, alpha=0.4, color=col[2])
    ax[1, 1].set_title(f"(d) Room 2 canonical variates (rank)\n"
                       f"energy corr(a,v)={np.corrcoef(a,v)[0,1]:.2f}  "
                       f"canonical corr={d['r']:.2f}")
    ax[1, 1].set_xlabel("audio variate (rank)"); ax[1, 1].set_ylabel("video variate (rank)")

    fig.suptitle("Cross-modal coherence is statistically real and lives in the multivariate "
                 "structure,\nnot the energy scalars", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_coherence_proof.png", dpi=600); plt.close(fig)
    print(f"\nWrote coherence_proof.csv, fig_coherence_proof.png in {OUT}")


if __name__ == "__main__":
    main()
