"""Three-panel environmental trajectory figure for Room 2.

Run: python src/viz/plot_env_trajectory.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ENV_FEATURES_ROOM2

# ---------------------------------------------------------------------------
ENV_CSV   = str(ENV_FEATURES_ROOM2)
OUT_PNG   = "env_trajectory_annotated_Room2.png"
ROLL      = 7          # days, smoothing window for trend lines
ROOM      = "Room 2"
# ---------------------------------------------------------------------------

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ENV_CSV)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # rolling trends (in case they aren't already in the file)
    if "temp_roll_mean_c" not in df:
        df["temp_roll_mean_c"] = df["temp_day_mean_c"].rolling(ROLL, center=True, min_periods=1).mean()
    if "rh_roll_mean_pct" not in df:
        df["rh_roll_mean_pct"] = df["rh_day_mean_pct"].rolling(ROLL, center=True, min_periods=1).mean()
    df["am_range_roll"] = df["temp_am_range_c"].rolling(ROLL, center=True, min_periods=1).mean()

    d = df["date"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- Panel 1: Temperature ------------------------------------------------
    ax1.fill_between(d, df["temp_am_min_c"], df["temp_am_max_c"],
                     color="#d46", alpha=0.12, label="AM min–max band")
    ax1.plot(d, df["temp_day_mean_c"], color="#d46", lw=0.7, alpha=0.5, label="daily mean")
    ax1.plot(d, df["temp_roll_mean_c"], color="#901", lw=2.2, label=f"{ROLL}-day trend")

    # brooding phase shading (first ~4 weeks from placement)
    brood_end = d.iloc[0] + pd.Timedelta(days=28)
    ax1.axvspan(d.iloc[0], brood_end, color="#f4a", alpha=0.06)
    ax1.annotate("Brooding set-point step-down\n(~first 4 weeks)",
                 xy=(d.iloc[0] + pd.Timedelta(days=8), 30),
                 fontsize=9, color="#901")
    ax1.annotate("Flat plateau (~23.8–24.2 °C):\nenv carries little slow-band info here",
                 xy=(brood_end + pd.Timedelta(days=25),
                     df["temp_day_mean_c"].iloc[-40:].mean() + 1.6),
                 fontsize=9, color="#555")
    ax1.set_ylabel("Temp (°C)")
    ax1.set_title(f"Environmental slow trajectory — {ROOM} (single production cycle)")
    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    ax1.grid(alpha=0.2)

    # --- Panel 2: Relative humidity -----------------------------------------
    ax2.plot(d, df["rh_day_mean_pct"], color="#37a", lw=0.7, alpha=0.5, label="daily mean")
    ax2.plot(d, df["rh_roll_mean_pct"], color="#036", lw=2.2, label=f"{ROLL}-day trend")

    # mark the two regime shifts by detecting the largest rolling-mean jumps
    rh = df["rh_roll_mean_pct"]
    jumps = rh.diff().abs()
    # find a rise (positive) and a crash (negative) as the two biggest signed moves
    rise_i = rh.diff().idxmax()
    crash_i = rh.diff().idxmin()
    for idx, txt, col in [(rise_i, "RH regime rise\n(exogenous)", "#093"),
                          (crash_i, "RH regime crash\n(exogenous)", "#a30")]:
        if pd.notna(idx):
            ax2.axvline(d.iloc[idx], color=col, ls="--", lw=1.3, alpha=0.8)
            ax2.annotate(txt, xy=(d.iloc[idx], rh.iloc[idx]),
                         xytext=(8, 0), textcoords="offset points",
                         fontsize=8, color=col, va="center")
    ax2.set_ylabel("RH (%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    # --- Panel 3: AM range widening -----------------------------------------
    ax3.plot(d, df["temp_am_range_c"], color="#7a3", lw=0.7, alpha=0.4, label="AM min–max spread (daily)")
    ax3.plot(d, df["am_range_roll"], color="#463", lw=2.2, label=f"{ROLL}-day trend")
    ax3.annotate("AM spread widens with age\n(slow signal the flat mean hides)",
                 xy=(d.iloc[int(len(d) * 0.6)], df["am_range_roll"].iloc[int(len(d) * 0.6)] + 0.8),
                 fontsize=9, color="#463")
    ax3.set_ylabel("AM range (°C)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.2)

    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    out_png = os.path.expanduser(OUT_PNG)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=600)
    print(f"Saved -> {out_png}")
    print(f"RH rise marked at  : {d.iloc[rise_i].date()}")
    print(f"RH crash marked at : {d.iloc[crash_i].date()}")


if __name__ == "__main__":
    main()