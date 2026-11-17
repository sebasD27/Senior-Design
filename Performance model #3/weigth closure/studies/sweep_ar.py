# Aspect-ratio sweep: closed design vs wing AR, everything else from inputs.py (engine, dash altitude),
# weight cap relaxed to 150 lb. The solves step outward from the inputs.py AR in both directions, each
# starting from its neighbour's solution.
#   python studies/sweep_ar.py                 # AR 1.0 - 4.0 in steps of 0.1
#   python studies/sweep_ar.py 1.2 3.0 0.05
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import INK, INK_2, SERIES, W_CAP, W_LIMIT, PLOT_DIR, baseline, plt, save, solve, style
import inputs
from missionsweep import max_sweep_deg, wing_planform


def walk(args):
    """Solve a list of AR values in order, each from the previous closed solution."""
    ars, h, engine = args
    out, seed = {}, None
    for ar in ars:
        m = baseline(AR=float(ar))
        _, _, _, r, _ = solve(("", h, m, engine) + ((seed,) if seed else ()))
        out[ar] = r
        seed = r or seed
    return out


if __name__ == "__main__":
    lo, hi, step = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (1.0, 4.0, 0.1)
    ars = [round(float(a), 4) for a in np.arange(lo, hi + step / 2, step)]
    h = inputs.ALTITUDE if inputs.ALTITUDE is not None else 30000.0
    engine = inputs.ENGINE if inputs.ENGINE not in (None, "auto") else "P650-PRO-S-R"
    ar0 = baseline().AR
    up = [a for a in ars if a >= ar0]
    down = sorted([a for a in ars if a < ar0], reverse=True)
    with ProcessPoolExecutor(2) as pool:
        table = {}
        for part in pool.map(walk, [(up, h, engine), (down, h, engine)]):
            table.update(part)

    rows = []
    for ar in ars:
        r = table.get(ar)
        if r is None:
            rows.append((ar, None))
            continue
        p = wing_planform(r["S"], ar, r["sweep_deg"])
        rows.append((ar, dict(W=r["W_TO"], D=r["D"], S=r["S"], span=p["span"] * 12, sweep=r["sweep_deg"],
                              limit=max_sweep_deg(baseline(AR=ar)), croot=p["c_root"] * 12, W_wing=r["W_wing"])))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "sweep_ar.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AR", "W_TO_lb", "drag_lbf", "S_ft2", "span_in", "root_chord_in", "LE_sweep_deg",
                    "delta_limit_deg", "W_wing_lb"])   # drag is constant with a pinned engine
        for ar, d in rows:
            w.writerow([ar] + ([round(d[k], 4) for k in ("W", "D", "S", "span", "croot", "sweep", "limit", "W_wing")]
                               if d else [""] * 8))

    print(f"AR sweep, {engine} at {h:,.0f} ft, Mach {baseline().M:g} (cap {W_CAP:g} lb; '-' = no design)")
    print("    AR    W_TO   drag   S ft2  span in  root in  LE sweep (limit)  W_wing")
    for ar, d in rows:
        print(f"  {ar:4.2f}  " + (f"{d['W']:6.2f} {d['D']:6.1f}  {d['S']:5.2f}  {d['span']:6.1f}  {d['croot']:6.1f}"
                                   f"   {d['sweep']:5.1f} ({d['limit']:4.1f})     {d['W_wing']:5.2f}" if d else "-"))
    ok = [(d["W"], ar) for ar, d in rows if d]
    if ok:
        best_w, best_ar = min(ok)
        print(f"Lightest: AR {best_ar:g} at {best_w:.2f} lb")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    # (Mach 1 drag is not plotted: with a pinned engine it is set by the thrust margin and stays constant)
    panels = [("W", "Closed takeoff weight (lb)"), ("W_wing", "Wing weight (lb)"),
              ("span", "Wing span (in)"), ("sweep", "Leading-edge sweep (deg)")]
    xs = [ar for ar, d in rows if d]
    for ax, (key, label) in zip(axes.flat, panels):
        style(ax, label)
        ys = [d[key] for _, d in rows if d]
        ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=4,
                markeredgecolor="#fcfcfb", markeredgewidth=1)
        ax.axvline(ar0, color=INK_2, linestyle=":", linewidth=1)
        if key == "W":
            ax.axhline(W_LIMIT, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"{W_LIMIT:g} lb limit", (xs[0], W_LIMIT), xytext=(2, 3), textcoords="offset points",
                        fontsize=9, color=INK_2)
            ax.plot([best_ar], [best_w], marker="o", markersize=8, color=SERIES[0], markeredgecolor=INK)
            ax.annotate(f"lightest: AR {best_ar:g}, {best_w:.1f} lb", (best_ar, best_w), xytext=(8, 4),
                        textcoords="offset points", ha="left", va="bottom", fontsize=9, color=INK_2)
        if key == "sweep":
            ax.plot(xs, [d["limit"] for _, d in rows if d], color=INK_2, linestyle="--", linewidth=1)
            first = next(d for _, d in rows if d)
            ax.annotate("delta limit (tip chord 0)", (xs[0], first["limit"]), xytext=(6, -2),
                        textcoords="offset points", ha="left", va="top", fontsize=9, color=INK_2)
        ax.annotate(f"inputs.py AR {ar0:g}", (ar0, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    for ax in axes[1]:
        ax.set_xlabel("Wing aspect ratio")
    fig.suptitle(f"Aspect-ratio sweep: {engine}, Mach {baseline().M:g} dash at {h:,.0f} ft (inputs.py)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    print(f"Saved {save(fig, 'sweep_ar.png')} and sweep_ar.csv")
