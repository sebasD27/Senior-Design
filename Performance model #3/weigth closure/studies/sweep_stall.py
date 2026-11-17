# Landing trade: closed design vs stall speed, one curve per C_Lmax. The wing is sized by stall at landing
# weight (rail launch) or takeoff weight, so this is wing area against touchdown speed. Everything else
# from inputs.py (engine, dash altitude), weight cap relaxed to 150 lb. Each curve is solved outward from
# the inputs.py stall speed, every solve starting from its neighbour.
#   python studies/sweep_stall.py                  # 90 - 150 ft/s every 5 ft/s, C_Lmax 1.0 / 1.3 / 1.5
#   python studies/sweep_stall.py 80 160 5
import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import INK, INK_2, MARKERS, SERIES, W_CAP, W_LIMIT, PLOT_DIR, baseline, plt, save, solve, style
import inputs

CLMAX = [1.0, 1.3, 1.5]


def walk(args):
    """Solve the stall speeds in order for one C_Lmax, each from the previous closed solution."""
    clmax, speeds, h, engine = args
    out, seed = {}, None
    for v in speeds:
        _, _, _, r, _ = solve(("", h, baseline(V_stall=float(v), C_Lmax=clmax), engine) + ((seed,) if seed else ()))
        out[(clmax, v)] = r
        seed = r or seed
    return out


if __name__ == "__main__":
    lo, hi, step = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (90.0, 150.0, 5.0)
    speeds = [float(v) for v in np.arange(lo, hi + step / 2, step)]
    h = inputs.ALTITUDE if inputs.ALTITUDE is not None else 30000.0
    engine = inputs.ENGINE if inputs.ENGINE not in (None, "auto") else "P650-PRO-S-R"
    v0, cl0 = baseline().V_stall, baseline().C_Lmax
    jobs = []
    for cl in CLMAX:
        jobs.append((cl, [v for v in speeds if v >= v0], h, engine))
        jobs.append((cl, sorted([v for v in speeds if v < v0], reverse=True), h, engine))
    table = {}
    with ProcessPoolExecutor(len(jobs)) as pool:
        for part in pool.map(walk, jobs):
            table.update(part)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "sweep_stall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["C_Lmax", "V_stall_fps", "W_TO_lb", "S_ft2", "W_wing_lb", "drag_lbf", "landing_weight_lb"])
        for cl in CLMAX:
            for v in speeds:
                r = table[(cl, v)]
                w.writerow([cl, v] + ([round(r["W_TO"], 3), round(r["S"], 4), round(r["W_wing"], 3), round(r["D"], 2),
                                       round(r["W_ZF"] + r["W_f_res"], 3)] if r else [""] * 5))

    print(f"Stall-speed sweep, {engine} at {h:,.0f} ft (cap {W_CAP:g} lb; '-' = no design). W_TO lb / wing ft^2")
    print("  V_stall " + "".join(f"   C_Lmax {cl:<4g}   " for cl in CLMAX))
    for v in speeds:
        cells = []
        for cl in CLMAX:
            r = table[(cl, v)]
            cells.append(f"  {r['W_TO']:6.1f} / {r['S']:4.2f}   " if r else f"  {'-':>15s}   ")
        print(f"  {v:6.0f} " + "".join(cells))
    print("Lowest stall speed under 55 lb:")
    for cl in CLMAX:
        ok = [v for v in speeds if table[(cl, v)] and table[(cl, v)]["W_TO"] <= W_LIMIT]
        print(f"  C_Lmax {cl:g}: " + (f"{min(ok):.0f} ft/s ({table[(cl, min(ok))]['W_TO']:.1f} lb, "
                                      f"{min(ok) * 0.5925:.0f} kt)" if ok else "none"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (key, title) in zip(axes, [("W_TO", "Closed takeoff weight (lb)"), ("S", "Wing area (ft^2)")]):
        style(ax, title)
        ends = []
        for i, cl in enumerate(CLMAX):
            pts = [(v, table[(cl, v)][key]) for v in speeds if table[(cl, v)]]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=5, linewidth=2,
                    markeredgecolor="#fcfcfb", markeredgewidth=1, label=f"C_Lmax {cl:g}")
            ends.append([ys[0], xs[0], f"C_Lmax {cl:g}"])
        ends.sort()                      # direct labels at the low-speed end of each line, kept apart
        gap = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for y, x, name in ends:
            ax.annotate(name, (x, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=9, color=INK_2)
        ax.axvline(v0, color=INK_2, linestyle=":", linewidth=1)
        ax.annotate(f"inputs.py {v0:g} ft/s", (v0, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
        ax.set_xlabel("Stall speed at landing weight (ft/s)")
        ax.set_xlim(lo - 12, hi + 3)
        if key == "W_TO":
            ax.axhline(W_LIMIT, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"{W_LIMIT:g} lb limit", (lo - 10, W_LIMIT), xytext=(2, 4), textcoords="offset points",
                        ha="left", fontsize=9, color=INK_2)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Landing trade: stall speed and C_Lmax ({engine}, Mach {baseline().M:g} dash at {h:,.0f} ft)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    print(f"Saved {save(fig, 'sweep_stall.png')} and sweep_stall.csv")
