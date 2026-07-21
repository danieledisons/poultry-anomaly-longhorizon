#!/usr/bin/env python3
"""
Approach C, first build: GATED ONLINE COVARIANCE (the slow geometry as a gated state).

Motivation (from today's findings): the cross-modal / cross-band correlation
GEOMETRY rotates over the growth cycle (geometry_rotation/), so a fixed covariance
(#1) or fixed manifold (#2) yields a non-stationary null and late-cycle false
alarms; the rolling-scale stop-gap only patches the scale, not the geometry.

Here the slow state is the geometry itself: an online mean mu_t AND covariance
Sigma_t, both EWMA-updated but GATED by the trust weight alpha_t. During healthy
growth the geometry rotates slowly and the gate stays open, so Sigma_t TRACKS the
rotation and the operating Mahalanobis stays stationary (low, stable false alarms).
During a sustained departure the score rises, alpha_t -> 0, and BOTH mu and Sigma
freeze -- so the anomaly is held out of the geometry (no contamination) while
detection is preserved.

Prediction being tested: gated online covariance keeps late-cycle false-closure
LOW and STATIONARY (unlike fixed #1) while retaining injection detection.

Configs compared on the same causal data + injection:
  fixed          : covariance frozen at the reference (this is #1)
  online_ungated : covariance adapts every step (no gate) -> absorbs anomalies
  online_gated   : covariance adapts, gated (Approach C)

Integrity: causal init (reference only), ridge-regularised inverse each step,
physically-specified injection, data-driven deadband, honest false-alarm reporting.

Usage: python validation_new/approach_c/gated_online_covariance.py --spine <csv> --room 2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
spec = importlib.util.spec_from_file_location("ra", HERE.parent / "gate_ablation" / "gate_ablation_richaudio.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
TREND_WIN = ra.TREND_WIN

LAM_MU = 0.10          # mean EWMA rate
LAM_COV = 0.02         # covariance EWMA rate (slower -> geometry is the slow state)
RIDGE = 0.10           # diagonal loading for a stable inverse
GAMMA = 0.85; C_CAP = 4.0; THETA_CLOSE = 6.0; THETA_OPEN = 2.0; B_DEAD = 1.0
INJ_BANDS = list(range(20, 41)); INJ_MAG = 4.0; INJ_LEN = 72; REF_FRAC = 0.35


def inv_ridge(S):
    d = S.shape[0]
    return np.linalg.inv(S + RIDGE * (np.trace(S) / d) * np.eye(d))


HUBER_C = 2.5          # robust update: points beyond med + HUBER_C*sig are down-weighted (no lock-in)
ALPHA_FLOOR = 0.12     # gated_floor: geometry never fully freezes -> tracks slow rotation, no lock-in
MEAN_FLOOR = 0.0       # but the MEAN can freeze fully (mean contamination is the real threat)
BETA_VEL = 0.02        # predictive: EWMA rate for the geometry-velocity (age-derivative) estimate


def psd_clip(S):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-6, None)
    return (V * w) @ V.T


def run(Z, mu0, S0, med, sig, mode):
    """mode in {fixed, online_ungated, online_gated, robust, gated_floor,
    meangate_covfree, directional}. Returns score, weight."""
    n, d = Z.shape
    mu = mu0.copy(); S = S0.copy(); Sinv = inv_ridge(S)
    score = np.zeros(n); wt = np.ones(n); P = 0.0
    uhat = np.zeros(d)                                  # estimated anomaly direction
    vmu = np.zeros(d); vS = np.zeros((d, d))            # geometry velocity (age-derivative)
    for t in range(n):
        r = Z[t] - mu
        score[t] = np.sqrt(max(r @ Sinv @ r, 0.0))
        if mode == "online_gated":
            # hard hysteresis gate (freezes geometry; suffers lock-in)
            e = min(max((score[t] - med) / sig - B_DEAD, 0.0), C_CAP)
            P = GAMMA * P + e
            a = 0.0 if P >= THETA_CLOSE else (1.0 if P <= THETA_OPEN else
                 (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN))
        elif mode == "robust":
            # SOFT per-step Huber trust, NO hysteresis lock-in.
            thr = med + HUBER_C * sig
            a = 1.0 if score[t] <= thr else float(thr / score[t])
        elif mode in ("gated_floor", "meangate_covfree", "directional", "predictive", "hybrid", "stable_directional"):
            e = min(max((score[t] - med) / sig - B_DEAD, 0.0), C_CAP)
            P = GAMMA * P + e
            a = 0.0 if P >= THETA_CLOSE else (1.0 if P <= THETA_OPEN else
                 (THETA_CLOSE - P) / (THETA_CLOSE - THETA_OPEN))
        else:
            a = 1.0

        if mode == "stable_directional":
            # The contribution, stabilised. Covariance ALWAYS adapts (tracks rotation,
            # stationary null). The anomaly direction is protected ONLY once the gate
            # is CONFIRMED closed (a~0, a sustained departure) -- not per-step -- using
            # a slowly-accumulated, stable direction estimate. In that state the mean
            # freezes (no absorption) and the covariance update drops the anomaly
            # component (rr_perp), so the anomaly axis keeps its normal variance and
            # detection is preserved; every other direction keeps tracking the rotation.
            confirmed = a < 0.05
            if confirmed:
                nu = r / (np.linalg.norm(r) + 1e-9)
                uhat = 0.9 * uhat + 0.1 * nu
            else:
                uhat = 0.8 * uhat
            a_mu = a                                     # mean freezes when confirmed
            mu_new = mu + LAM_MU * a_mu * r
            rr = Z[t] - mu_new
            un = uhat / (np.linalg.norm(uhat) + 1e-9)
            if confirmed and np.linalg.norm(uhat) > 1e-6:
                rr = rr - (rr @ un) * un                 # protect the anomaly axis
            S = psd_clip((1 - LAM_COV) * S + LAM_COV * np.outer(rr, rr))
            mu = mu_new; Sinv = inv_ridge(S); wt[t] = a
            continue

        if mode == "hybrid":
            # STABLE memory variant: predict the MEAN along its learned age-velocity
            # (breaks mean-contamination without unstable covariance integration), and
            # track the COVARIANCE with a robust down-weight (tracks rotation, resists
            # anomalous inflation). Mean velocity is learned from trusted steps only.
            thr = med + HUBER_C * sig
            w_rob = 1.0 if score[t] <= thr else float(thr / score[t])
            dmu_assim = LAM_MU * r
            mu_new = mu + a * dmu_assim + (1 - a) * vmu
            rr = Z[t] - mu_new
            S = psd_clip((1 - LAM_COV * w_rob) * S + LAM_COV * w_rob * np.outer(rr, rr))
            if a > 0.8:
                vmu = (1 - BETA_VEL) * vmu + BETA_VEL * (mu_new - mu)
            mu = mu_new; Sinv = inv_ridge(S); wt[t] = a
            continue

        if mode == "predictive":
            # AGE-CONDITIONED EXPECTED-GEOMETRY PRIOR.
            # Trusted (gate-open) steps assimilate AND teach the geometry velocity
            # (its age-derivative). During gate closure the geometry ADVANCES along
            # that learned velocity instead of freezing (avoids lock-in drift) or
            # absorbing the anomaly (residual never enters the update). The anomaly
            # thus stays far from the predicted-normal geometry -> detected.
            dmu_assim = LAM_MU * r
            mu_new = mu + a * dmu_assim + (1 - a) * vmu
            rr = Z[t] - mu_new
            dS_assim = LAM_COV * (np.outer(rr, rr) - S)
            S_new = psd_clip(S + a * dS_assim + (1 - a) * vS)
            if a > 0.8:                                  # learn velocity only from trusted steps
                vmu = (1 - BETA_VEL) * vmu + BETA_VEL * (mu_new - mu)
                vS = (1 - BETA_VEL) * vS + BETA_VEL * (S_new - S)
            mu = mu_new; S = S_new; Sinv = inv_ridge(S); wt[t] = a
            continue
        wt[t] = a
        a_mu = a                                       # mean trust (can freeze)
        if mode == "gated_floor":
            a_cov = max(a, ALPHA_FLOOR)
        elif mode in ("meangate_covfree", "directional"):
            a_cov = 1.0                                # covariance keeps adapting...
        else:
            a_cov = a
        mu_new = mu + LAM_MU * a_mu * r
        if mode != "fixed":
            rr = (Z[t] - mu_new)
            if mode == "directional":
                # ...but in the anomaly direction it must NOT adapt (else it absorbs
                # the departure). Track the anomaly direction when the gate is closing
                # and PROJECT IT OUT of the covariance update, so Sigma tracks the
                # natural rotation in every OTHER direction while the anomaly axis
                # keeps its normal (small) variance -> detection preserved.
                if a < 0.999:
                    nu = r / (np.linalg.norm(r) + 1e-9)
                    uhat = 0.5 * uhat + 0.5 * nu
                    un = uhat / (np.linalg.norm(uhat) + 1e-9)
                    rr = rr - (rr @ un) * un            # remove anomaly component
                else:
                    uhat = 0.9 * uhat                   # decay the estimate when normal
            S = (1 - LAM_COV * a_cov) * S + LAM_COV * a_cov * np.outer(rr, rr)
            Sinv = inv_ridge(S)
        mu = mu_new
    return score, wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default=str(DATA / "spine_room2_rich.csv"))
    ap.add_argument("--room", default="2"); ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--inj_type", default="step", choices=["step", "ramp"],
                    help="step = sudden sustained; ramp = slow-onset (the discriminating test)")
    a = ap.parse_args(); R = a.room; ITYPE = a.inj_type

    df = pd.read_csv(a.spine).sort_values("time").reset_index(drop=True)
    mel = [c for c in df.columns if c.startswith("aud_mel") and c.endswith("_mean")]
    df = df[df[mel].notna().all(axis=1)].reset_index(drop=True); n = len(df)
    Z = ra.causal_slow_band(df, mel)
    ref_end = int(REF_FRAC * n)
    mu0 = Z[:ref_end][:TREND_WIN].mean(0)
    S0, lam = ra.ledoit_wolf(Z[:ref_end])

    # calibrate deadband on the ONLINE-GATED operating score over the clean reference
    op, _ = run(Z[:ref_end], mu0, S0, 0.0, 1.0, "online_ungated")
    op = op[TREND_WIN:]
    med = np.median(op); sig = 1.4826 * np.median(np.abs(op - med)) or op.std()
    det = np.quantile(op, 0.99)

    # injection
    start = ref_end + 150; end = min(start + INJ_LEN, n)
    bidx = [mel.index(f"aud_mel{b:02d}_mean") for b in INJ_BANDS if f"aud_mel{b:02d}_mean" in mel]
    bmad = 1.4826 * np.median(np.abs(Z[:ref_end][:, bidx] - np.median(Z[:ref_end][:, bidx], 0)), 0)
    Zin = Z.copy()
    if ITYPE == "step":
        Zin[start:end][:, bidx] += INJ_MAG * bmad                       # sudden sustained
    else:
        ramp = np.linspace(0, INJ_MAG, end - start)[:, None]           # slow-onset ramp
        Zin[start:end][:, bidx] += ramp * bmad

    def metrics(mode):
        sc_i, al_i = run(Zin, mu0, S0, med, sig, mode)     # injected
        sc_c, al_c = run(Z, mu0, S0, med, sig, mode)       # clean
        # detection threshold: rolling? here fixed det (should hold if score stationary)
        seg = sc_i[start:end]
        retention = float((seg >= det).mean())
        below = np.where(seg < det)[0]; lat = int(below[0]) if len(below) else np.inf
        late = np.ones(n, bool); late[:ref_end] = False; late[start:end] = False
        # false-alarm = clean operating score exceeding det (a would-be alarm)
        clean_fa = float((sc_c[late] >= det).mean())
        # drift of the operating level (stationarity): median late vs early
        early = np.median(sc_c[ref_end:ref_end + 300]); latev = np.median(sc_c[-300:])
        return dict(retention=round(retention, 3), latency=lat,
                    late_false_alarm=round(clean_fa, 3),
                    op_early=round(float(early), 2), op_late=round(float(latev), 2),
                    sc_i=sc_i, sc_c=sc_c, al_i=al_i)

    res = {m: metrics(m) for m in ["fixed", "online_ungated", "online_gated", "robust", "gated_floor", "meangate_covfree", "directional", "predictive", "hybrid", "stable_directional"]}

    summ = pd.DataFrame([{"room": R, "config": m, "shrink_lam": round(lam, 3),
        "det_thresh": round(det, 2),
        "inj_retention": res[m]["retention"], "inj_latency_hr": res[m]["latency"],
        "late_false_alarm": res[m]["late_false_alarm"],
        "clean_op_early": res[m]["op_early"], "clean_op_late": res[m]["op_late"]}
        for m in res])
    summ.to_csv(HERE / "csv" / f"approachc_room{R}_{ITYPE}_summary.csv", index=False)

    # ---- figure ----
    x = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    ax[0].axhline(det, color="k", ls="--", lw=0.8, label="detect thresh")
    ax[0].plot(x, res["online_gated"]["sc_c"], color="tab:green", lw=0.6, alpha=0.7, label="naive gated — clean (drifts up)")
    ax[0].plot(x, res["robust"]["sc_c"], color="tab:purple", lw=0.7, label="robust (C) — clean (stationary)")
    ax[0].axvspan(0, ref_end, color="blue", alpha=0.05, label="reference")
    ax[0].set_ylabel("clean operating Mahalanobis"); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title(f"Room {R}: ROBUST online covariance (C) — clean FA {res['robust']['late_false_alarm']} & "
                    f"injection retention {res['robust']['retention']} (naive gate FA {res['online_gated']['late_false_alarm']})")
    ax[0].set_ylim(0, np.percentile(res["online_gated"]["sc_c"], 99))
    ax[1].axhline(det, color="k", ls="--", lw=0.8)
    ax[1].plot(x, res["robust"]["sc_i"], color="tab:purple", lw=0.8, label="robust (C) — injected (holds)")
    ax[1].plot(x, res["online_ungated"]["sc_i"], color="tab:orange", lw=0.8, alpha=0.8, label="ungated — injected (absorbs)")
    ax[1].axvspan(start, end, color="orange", alpha=0.15, label="injection")
    ax[1].set_ylabel("injected Mahalanobis"); ax[1].set_xlabel("audio hours"); ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_ylim(0, np.percentile(res["robust"]["sc_i"], 99.5))
    fig.tight_layout(); fig.savefig(HERE / "figs" / f"fig_approachc_room{R}_{ITYPE}.png", dpi=140)

    print(f"=== Room {R} Approach C: gated online covariance ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
