#!/usr/bin/env python3
"""Event-evaluation harness for the development-conditioned coherence detector. Fits the
clean model once (age-energy channel + CCA coherence channel), then scores ANY event given
as feature-level residual streams -- synthetic today, real tomorrow -- and reports whether
it fired, when, on which channel, and the inferred cause (magnitude vs incoherence). Also
runs a synthetic validation so we can see the approach works before the real events arrive.

Real events plug in via a CSV with columns: time, is_event(0/1), resid_aud_*, resid_vid_*.

Run: python src/pipeline/dev_event_harness.py --residuals results/dev_conditioned/model/residuals_room2.csv
     python src/pipeline/dev_event_harness.py --residuals ... --event my_real_event.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RESULTS_DIR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

OUT = RESULTS_DIR / "dev_conditioned" / "events"
MODEL = RESULTS_DIR / "dev_conditioned" / "model"
GATE = dict(deadband=1.0, decay=0.85, open=2.0, cap=2.0)


def _zpar(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return med, mad


# ---------------------------------------------------------------- the detector
class CoherenceDetector:
    """Learns the clean age-energy + cross-modal coherence model, then scores any
    feature-level (audio, video) residual stream. This is the single interface that
    both synthetic and real events flow through."""

    def fit(self, A, V, k=5):
        self.mA, self.sA = A.mean(0), A.std(0) + 1e-9
        self.mV, self.sV = V.mean(0), V.std(0) + 1e-9
        self.pca_a = PCA(k).fit((A - self.mA) / self.sA)
        self.pca_v = PCA(k).fit((V - self.mV) / self.sV)
        Ap = self.pca_a.transform((A - self.mA) / self.sA)
        Vp = self.pca_v.transform((V - self.mV) / self.sV)
        self.cca = CCA(1).fit(Ap, Vp)
        u, w = self._uw(A, V)
        self.b = np.polyfit(u, w, 1)[0]
        self.umed, _ = _zpar(u); self.wmed, _ = _zpar(w)
        self.dmed, self.dmad = _zpar(w - self.b * u)
        self.aMed, self.aMad = _zpar(np.linalg.norm(A, axis=1))
        self.vMed, self.vMad = _zpar(np.linalg.norm(V, axis=1))
        # keep clean pool for synthetic timelines + incoherence swaps
        self._A, self._V, self._w = A, V, w
        return self

    def _uw(self, A, V):
        Ap = self.pca_a.transform((A - self.mA) / self.sA)
        Vp = self.pca_v.transform((V - self.mV) / self.sV)
        U, W = self.cca.transform(Ap, Vp)
        return U[:, 0], W[:, 0]

    def channels(self, A, V):
        """Return (age_energy, coherence) per-hour scores for any feature-level input."""
        a = (np.linalg.norm(A, axis=1) - self.aMed) / self.aMad
        v = (np.linalg.norm(V, axis=1) - self.vMed) / self.vMad
        u, w = self._uw(A, V)
        coh = np.abs((w - self.b * u) - self.dmed) / self.dmad
        return np.maximum(a, v), coh

    def score(self, A, V):
        age, coh = self.channels(A, V)
        return dict(age=age, coherence=coh, full=np.maximum(age, coh))


# ---------------------------------------------------------------- gate
def gate(score, close):
    P = 0.0; latched = False; out = np.zeros(len(score), bool)
    for t, s in enumerate(score):
        e = min(max(0.0, s - GATE["deadband"]), GATE["cap"])
        P = GATE["decay"] * P + e
        latched = (P > GATE["open"]) if latched else (P >= close)
        out[t] = latched
    return out

def onsets(level):
    o = np.zeros(len(level), bool); o[1:] = level[1:] & ~level[:-1]; o[0] = level[0]
    return o


# ---------------------------------------------------------------- feature-level events
def make_concordant(det, rng, timeline=240, mag=2.5, dur=(12, 30)):
    """Both modalities scale up together (magnitude anomaly). Feature-level."""
    ia = rng.integers(0, len(det._A), timeline); iv = ia.copy()
    A, V = det._A[ia].copy(), det._V[iv].copy()
    a0 = rng.integers(40, timeline - 40); L = int(rng.integers(*dur)); sl = slice(a0, a0 + L)
    A[sl] *= (1 + mag); V[sl] *= (1 + mag)
    return A, V, (a0, a0 + L)

def make_incoherence(det, rng, timeline=240, dur=(12, 30)):
    """Each modality individually normal, but the audio-video pairing is broken: during
    the event, video is replaced by a REAL clean video vector chosen to sit far from what
    the concurrent audio predicts. Marginals stay in-distribution; only the coupling breaks.
    This is the feature-level proxy for a real incoherence event."""
    ia = rng.integers(0, len(det._A), timeline)
    A, V = det._A[ia].copy(), det._V[ia].copy()
    a0 = rng.integers(40, timeline - 40); L = int(rng.integers(*dur)); sl = slice(a0, a0 + L)
    u, _ = det._uw(A[sl], V[sl]); w_hat = det.b * u
    for j, t in enumerate(range(a0, a0 + L)):
        far = np.argmax(np.abs(det._w - w_hat[j]))    # clean video row least predicted by audio
        V[t] = det._V[far]
    return A, V, (a0, a0 + L)


# ---------------------------------------------------------------- evaluate one event
def evaluate_event(det, A, V, span, close):
    """Return detection outcome + cause attribution for a single event (real or synthetic)."""
    sc = det.score(A, V)
    fired = {k: onsets(gate(sc[k], close[k])) for k in sc}
    a0, a1 = span
    out = {}
    for k in sc:
        hit = fired[k][a0:a1]
        out[k] = dict(detected=bool(hit.any()),
                      delay_h=int(np.argmax(hit)) if hit.any() else None)
    # cause attribution: which channel carries the signal inside the event window
    age_pk = sc["age"][a0:a1].max(); coh_pk = sc["coherence"][a0:a1].max()
    cause = "magnitude (concordant)" if age_pk > coh_pk else "incoherence (cross-modal)"
    out["attribution"] = dict(age_peak=round(float(age_pk), 2),
                              coherence_peak=round(float(coh_pk), 2), inferred_cause=cause)
    return out


def calibrate(det, rng, target_fa=0.3, timeline=240, n=200):
    close = {}
    clean = [make_concordant(det, rng, timeline, mag=0.0)[:2] for _ in range(n)]
    for k in ("age", "coherence", "full"):
        grid = np.linspace(0.5, 12, 60); best, gap = grid[-1], 1e9
        for h in grid:
            fa = np.mean([onsets(gate(det.score(A, V)[k], h)).sum() for A, V in clean])
            if abs(fa - target_fa) < gap:
                gap, best = abs(fa - target_fa), h
        close[k] = best
    return close


# ---------------------------------------------------------------- synthetic validation
def run_synthetic(det, rng, close, trials=400):
    rows = []
    for kind, gen in [("concordant", make_concordant), ("incoherence", make_incoherence)]:
        rec = {"age": 0, "coherence": 0, "full": 0}; correct_cause = 0
        for _ in range(trials):
            A, V, span = gen(det, rng)
            r = evaluate_event(det, A, V, span, close)
            for k in rec:
                rec[k] += r[k]["detected"]
            want = "magnitude" if kind == "concordant" else "incoherence"
            correct_cause += want in r["attribution"]["inferred_cause"]
        rows.append(dict(event_type=kind, n=trials,
                         recall_age=round(rec["age"]/trials, 3),
                         recall_coherence=round(rec["coherence"]/trials, 3),
                         recall_full=round(rec["full"]/trials, 3),
                         cause_accuracy=round(correct_cause/trials, 3)))
    return pd.DataFrame(rows)


def load_residuals(path):
    df = pd.read_csv(path)
    ac = [c for c in df.columns if c.startswith("resid_aud_")]
    vc = [c for c in df.columns if c.startswith("resid_vid_")]
    A = df[ac].to_numpy(float); V = df[vc].to_numpy(float)
    k = ~(np.isnan(A).any(1) | np.isnan(V).any(1))
    return A[k], V[k], df, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", default=str(MODEL / "residuals_room2.csv"),
                    help="clean residuals used to FIT the detector")
    ap.add_argument("--event", default=None,
                    help="optional real-event CSV (time,is_event,resid_aud_*,resid_vid_*)")
    ap.add_argument("--trials", type=int, default=400)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    A, V, _, _ = load_residuals(a.residuals)
    det = CoherenceDetector().fit(A, V)
    close = calibrate(det, rng)
    print("matched-FA thresholds: " + ", ".join(f"{k}={v:.2f}" for k, v in close.items()))

    val = run_synthetic(det, rng, close, a.trials)
    val.to_csv(OUT / "synthetic_validation.csv", index=False)
    print("\n=== Synthetic validation (feature-level events, matched FA) ===")
    print(val.to_string(index=False))

    if a.event:
        Ae, Ve, dfe, keep = load_residuals(a.event)
        span_mask = dfe.loc[keep, "is_event"].to_numpy().astype(bool) \
            if "is_event" in dfe.columns else np.zeros(len(Ae), bool)
        if span_mask.any():
            idx = np.where(span_mask)[0]; span = (idx.min(), idx.max() + 1)
        else:
            span = (0, len(Ae))
        r = evaluate_event(det, Ae, Ve, span, close)
        print("\n=== REAL EVENT ===")
        for k in ("age", "coherence", "full"):
            print(f"  {k:10s}: detected={r[k]['detected']}  delay_h={r[k]['delay_h']}")
        print(f"  attribution: {r['attribution']}")

    print(f"\nWrote synthetic_validation.csv in {OUT}")


if __name__ == "__main__":
    main()
