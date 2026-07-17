"""
alpha_gate.py — Minimal residual-gated trust mechanism (alpha_t)

Core dissertation contribution, kept deliberately minimal and testable:
a STATEFUL FUNCTION, not a learned model.

Two pieces:
  1. slow_fast_decompose(): split an hourly signal into slow band
     (gap-aware rolling-median trend + smoothed diurnal cycle) and a
     fast-band residual. Matches the Room 2 findings-note methodology.
  2. AlphaGate: a leaky-bucket persistence accumulator over the fast
     residual. The bucket fills under sustained unexplained residual
     energy and drains under ordinary noise. alpha_t (the "trust" /
     learning-rate gate) CLOSES when persistence crosses a threshold
     and REOPENS when residual energy subsides.

Design intent: the gate responds to PERSISTENCE of unexplained energy,
not to instantaneous magnitude. A brief, large, self-resolving spike
should NOT close the gate; a smaller-but-sustained departure SHOULD.
This is the property the synthetic injection test validates.

The SAME parameters must be reused across rooms — no per-room retuning —
or the generalization test is meaningless.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Slow / fast decomposition (gap-aware)
# ----------------------------------------------------------------------
def slow_fast_decompose(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    slow_window_days: int = 7,
    diurnal_smooth: int = 3,
):
    """Decompose an irregularly-sampled hourly series into slow + fast bands.

    slow band  = rolling-median trend (gap-aware, time-based window)
                 + smoothed diurnal (mean-by-hour-of-day on detrended residual,
                   circular-smoothed so 23:00 and 00:00 are neighbours)
    fast band  = value - slow band  (the residual the gate consumes)

    Rows with NaN value_col are preserved but produce NaN residual.
    """
    d = df[[time_col, value_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    s = d.set_index(time_col)[value_col]

    # --- slow trend: time-based rolling median, gap-aware, centered ---
    win = f"{slow_window_days}D"
    trend = s.rolling(win, center=True, min_periods=3).median()
    trend = trend.interpolate(method="time", limit_direction="both")

    detrended = s - trend

    # --- diurnal component: mean by hour-of-day, circular-smoothed ---
    hod = detrended.index.hour
    by_hour = pd.Series(detrended.values, index=hod).groupby(level=0).mean()
    by_hour = by_hour.reindex(range(24))
    # fill any empty hours by interpolation before smoothing
    by_hour = by_hour.interpolate().bfill().ffill()
    # circular smoothing
    ext = pd.concat([by_hour, by_hour, by_hour])
    sm = ext.rolling(diurnal_smooth, center=True, min_periods=1).mean()
    diurnal = sm.iloc[24:48]
    diurnal.index = range(24)

    diurnal_full = pd.Series(diurnal.reindex(hod).values, index=s.index)
    slow = trend + diurnal_full
    fast = s - slow

    out = pd.DataFrame(
        {
            time_col: s.index,
            value_col: s.values,
            "slow": slow.values,
            "trend": trend.values,
            "diurnal": diurnal_full.values,
            "fast_residual": fast.values,
        }
    )
    return out, diurnal


# ----------------------------------------------------------------------
# The alpha_t trust gate (leaky-bucket persistence accumulator)
# ----------------------------------------------------------------------
class AlphaGate:
    """Residual-gated trust mechanism.

    State variable P_t (persistence bucket) evolves as:

        e_t = clip(max(0, |r_t|/scale - deadband), 0, per_step_cap)
        P_t = decay * P_{t-1} + e_t               # leaky accumulation

    The per_step_cap is essential: it bounds how much a SINGLE hour can add
    to the bucket, so no lone spike — however large — can overflow the gate
    on its own. Closure then requires energy SUSTAINED across several hours.
    Without the cap the gate degrades into a magnitude detector.

    Gate:
        closes (alpha_t -> 0) when P_t >= close_threshold
        reopens (alpha_t -> 1) when P_t <= open_threshold   (hysteresis)

    alpha_t is emitted as a soft value in [0, 1] via a linear ramp between
    open_threshold and close_threshold, with a hard latch for the closed
    state so brief dips don't chatter the gate open.

    Parameters are LOCKED and shared across rooms.
    """

    def __init__(
        self,
        scale: float = 1.0,          # residual normaliser (robust std of clean residual)
        deadband: float = 1.0,       # ignore residual energy below this many 'scales'
        decay: float = 0.85,         # bucket leak per hour (0.85 => ~ few-hour memory)
        close_threshold: float = 6.0,
        open_threshold: float = 2.0,  # hysteresis: must fall well below close to reopen
        per_step_cap: float = 2.0,    # max energy a single hour can add to the bucket
    ):
        assert open_threshold < close_threshold
        self.scale = scale
        self.deadband = deadband
        self.decay = decay
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.per_step_cap = per_step_cap

    def run(self, residual: np.ndarray):
        """Run the gate over a residual stream. NaNs are treated as 0 energy
        (no information => no evidence to close on) but still leak the bucket.

        Returns dict of arrays: persistence, alpha (soft trust), closed (bool).
        """
        r = np.asarray(residual, dtype=float)
        n = len(r)
        P = np.zeros(n)
        alpha = np.ones(n)
        closed = np.zeros(n, dtype=bool)
        latched = False
        p = 0.0
        for t in range(n):
            rt = r[t]
            if np.isnan(rt):
                e = 0.0
            else:
                e = max(0.0, abs(rt) / self.scale - self.deadband)
                e = min(e, self.per_step_cap)   # bound single-hour contribution
            p = self.decay * p + e
            P[t] = p

            # hysteresis latch
            if latched:
                if p <= self.open_threshold:
                    latched = False
            else:
                if p >= self.close_threshold:
                    latched = True
            closed[t] = latched

            if latched:
                alpha[t] = 0.0
            else:
                # soft ramp between open and close thresholds
                frac = (p - self.open_threshold) / (
                    self.close_threshold - self.open_threshold
                )
                frac = min(max(frac, 0.0), 1.0)
                alpha[t] = 1.0 - frac  # 1 = fully trusting/learning, 0 = closed
        return {"persistence": P, "alpha": alpha, "closed": closed}


def robust_scale(x: np.ndarray) -> float:
    """MAD-based robust standard deviation, NaN-safe."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 1.0
    mad = np.median(np.abs(x - np.median(x)))
    return max(1.4826 * mad, 1e-6)