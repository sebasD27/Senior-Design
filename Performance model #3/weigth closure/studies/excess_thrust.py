# Excess thrust at the dash Mach number vs dash altitude, one curve per catalogue engine. Every point is
# its own airframe: the design closed (and optimized) for that engine at that dash altitude, the same
# continuation solves as weight_vs_altitude.py (q limit lifted, its altitude shaded). Only designs under
# the 55 lb limit are plotted. Excess is for steady level flight at the weight at the start of the dash:
#   excess thrust   = installed thrust (T_SLS x lapse x eta_inst) - drag            [lbf]
#   specific excess power P_s = excess thrust x V / W                               [ft/s]
# (the sizing holds a thrust margin over the whole acceleration path at takeoff weight, which is stricter)
#   python studies/excess_thrust.py                 # 0 - 36,000 ft every 1,000 ft
#   python studies/excess_thrust.py 10000 36000 1000
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import (ENGINES, INK_2, MARKERS, SERIES, W_LIMIT, PLOT_DIR, altitude_at_q, baseline, plt, save,
                     style)
from engine_choice import catalogue_engine
from mission_atmosphere import atmosphere_at_altitude
from mission_path import flight_point, point_drag
from weight_vs_altitude import descend


def excess(m, engine, h, r):
    """Steady Mach-M excess thrust of the closed design r at its dash altitude h."""
    p = flight_point(m, h, m.M * atmosphere_at_altitude(h).speed_of_sound_ft_s)
    W = r["W_cr_strt"]
    D = float(point_drag(m, p, weight=W, S=r["S"], tau=r["tau"], c_sw=r["cos_sw"], L=r["L_fus"], d=r["d_fus"]))
    T = catalogue_engine(engine).thrust * p.lapse * m.eta_inst
    return dict(h=h, W_TO=r["W_TO"], T=T, D=D, excess=T - D, Ps=(T - D) * p.speed / W, margin=T / D - 1.0)


if __name__ == "__main__":
    lo, hi, step = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (0.0, 36000.0, 1000.0)
    heights = [float(h) for h in np.arange(lo, hi + 1, step)]
    q_user = baseline().q_max
    m = baseline(q_max=math.inf)
    with ProcessPoolExecutor(max_workers=len(ENGINES)) as pool:
        solved = dict(pool.map(descend, [(e, heights, m) for e in ENGINES]))
    table = {e: [excess(m, e, h, solved[e][h]) for h in heights
                 if solved[e][h] and solved[e][h]["W_TO"] <= W_LIMIT] for e in ENGINES}

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "excess_thrust.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "altitude_ft", "W_TO_lb", "thrust_installed_lbf", "drag_lbf", "excess_thrust_lbf",
                    "Ps_ft_s", "thrust_margin"])
        for e in ENGINES:
            for row in table[e]:
                w.writerow([e, row["h"], round(row["W_TO"], 3), round(row["T"], 2), round(row["D"], 2),
                            round(row["excess"], 2), round(row["Ps"], 2), round(row["margin"], 4)])

    print(f"Excess thrust at Mach {m.M:g}, each engine's own closed design, designs under {W_LIMIT:g} lb only")
    print("  engine            altitude    W_TO   excess  margin     P_s")
    for e in ENGINES:
        if not table[e]:
            print(f"  {e:16s}  no design under {W_LIMIT:g} lb")
            continue
        for row in table[e]:
            print(f"  {e:16s} {row['h']:9,.0f}  {row['W_TO']:6.1f}  {row['excess']:6.1f}  {row['margin']:6.1%}"
                  f"  {row['Ps']:6.0f}")

    x0 = min((row["h"] for e in ENGINES for row in table[e]), default=lo) / 1000 - 4
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    panels = [("excess", f"Excess thrust at Mach {m.M:g} (lbf)"), ("Ps", f"Specific excess power at Mach {m.M:g} (ft/s)")]
    for ax, (key, title) in zip(axes, panels):
        style(ax, title)
        ends = []
        for i, e in enumerate(ENGINES):              # colors stay on their engine slots
            if not table[e]:
                continue
            xs = [row["h"] / 1000 for row in table[e]]
            ys = [row[key] for row in table[e]]
            ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=6, linewidth=2,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.0, label=e)
            ends.append([ys[0], xs[0], e.split("-")[0]])
        ax.set_ylim(bottom=0)
        ends.sort()                      # direct labels at the low-altitude end of each line, kept apart
        ylo, yhi = ax.get_ylim()
        gap = 0.04 * (yhi - ylo)
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for y, x, name in ends:
            ax.annotate(name, (x, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=9, color=INK_2)
        if math.isfinite(q_user):
            hq = altitude_at_q(q_user, m.M) / 1000
            ax.axvspan(x0, hq, color="#e4e3df", alpha=0.5, linewidth=0)
            ax.annotate(f"q > {q_user:g} psf\n(inputs.py q_max)", (hq, 0), xytext=(-6, 6),
                        textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK_2)
        ax.set_xlim(x0, hi / 1000 + 1)
        ax.set_xlabel("Dash altitude (1,000 ft)")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"Excess thrust of each engine's own closed design under {W_LIMIT:g} lb "
                 f"(steady Mach {m.M:g} at dash-start weight, {m.accel_margin:.0%} path margin in the sizing)",
                 x=0.01, ha="left", fontsize=12)
    print(f"Saved {save(fig, 'excess_thrust.png')} and excess_thrust.csv")
