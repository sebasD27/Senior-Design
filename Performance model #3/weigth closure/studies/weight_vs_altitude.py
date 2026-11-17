# Study 3: closed takeoff weight vs dash altitude, one curve per catalogue engine.
# How low can each engine dash, and what does a lower dash cost in weight? Everything else is inputs.py,
# with the weight cap relaxed to 150 lb and the dynamic-pressure limit lifted (its altitude is drawn).
#   python studies/weight_vs_altitude.py                    # 0 - 36,000 ft every 2,000 ft
#   python studies/weight_vs_altitude.py 10000 36000 1000
import csv
import math
import sys

import numpy as np

from _common import (ENGINES, INK_2, MARKERS, SERIES, W_CAP, W_LIMIT, PLOT_DIR, altitude_at_q, baseline,
                     plt, run_all, save, style)

if __name__ == "__main__":
    lo, hi, step = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (0.0, 36000.0, 2000.0)
    heights = [float(h) for h in np.arange(lo, hi + 1, step)]
    q_user = baseline().q_max
    m = baseline(q_max=math.inf)
    results = run_all([("", h, m, e) for e in ENGINES for h in heights])

    table = {e: {} for e in ENGINES}
    for _, h, e, r, _ in results:
        table[e][h] = r
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "weight_vs_altitude.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "altitude_ft", "W_TO_lb", "S_ft2", "L_fus_ft", "d_fus_in", "drag_lbf", "fuel_lb"])
        for e in ENGINES:
            for h in heights:
                r = table[e][h]
                w.writerow([e, h] + ([round(r["W_TO"], 3), round(r["S"], 4), round(r["L_fus"], 3),
                                      round(r["d_fus"] * 12, 2), round(r["D"], 2),
                                      round(r["W_TO"] - r["W_ZF"], 3)] if r else ["", "", "", "", "", ""]))

    print(f"Closed W_TO (lb) vs dash altitude, Mach {m.M:g}, cap {W_CAP:g} lb ('-' = no design under the cap)")
    print("  altitude " + "".join(f"{e.split('-')[0]:>9s}" for e in ENGINES))
    for h in heights:
        print(f"  {h:8,.0f} " + "".join(f"{table[e][h]['W_TO']:9.1f}" if table[e][h] else f"{'-':>9s}" for e in ENGINES))
    print("Lowest dash altitude under 55 lb:")
    for e in ENGINES:
        ok = [h for h in heights if table[e][h] and table[e][h]["W_TO"] <= W_LIMIT]
        print(f"  {e:16s} " + (f"{min(ok):,.0f} ft ({table[e][min(ok)]['W_TO']:.1f} lb)" if ok else "none"))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    style(ax, f"Closed takeoff weight vs dash altitude (Mach {m.M:g}, inputs.py, {m.accel_margin:.0%} thrust margin)")
    for i, e in enumerate(ENGINES):
        pts = [(h / 1000, table[e][h]["W_TO"]) for h in heights if table[e][h]]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=6, linewidth=2,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=e)
        ax.annotate(e.split("-")[0], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.axhline(W_LIMIT, color=INK_2, linestyle="--", linewidth=1)
    ax.annotate(f"{W_LIMIT:g} lb limit", (lo / 1000, W_LIMIT), xytext=(4, 4), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.set_xlim(lo / 1000 - 1, hi / 1000 + 2.5)
    if math.isfinite(q_user):
        hq = altitude_at_q(q_user, m.M) / 1000
        ax.axvspan(lo / 1000 - 1, hq, color="#e4e3df", alpha=0.5, linewidth=0)
        ax.annotate(f"q > {q_user:g} psf\n(inputs.py q_max)", (hq, ax.get_ylim()[1]), xytext=(-6, -6),
                    textcoords="offset points", ha="right", va="top", fontsize=9, color=INK_2)
    ax.set_xlabel("Dash altitude (1,000 ft)")
    ax.set_ylabel("Closed takeoff weight (lb)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    print(f"Saved {save(fig, 'weight_vs_altitude.png')} and weight_vs_altitude.csv")
