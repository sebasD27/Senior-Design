# Study 2: tornado chart. Each constant moved -20 % and +20 %, one at a time, from the inputs.py baseline
# at the inputs.py dash altitude (weight cap relaxed to 150 lb). The engine is a continuously scalable
# copy of the inputs.py engine (thrust in proportion to weight, same TSFC, catalogue diameter exponent), so
# the weight changes smoothly instead of jumping between catalogue sizes. Every solve starts from the
# baseline solution.
#   python studies/tornado.py
import csv

from _common import (INK, INK_2, SERIES, W_CAP, PLOT_DIR, baseline, plt, run_all, save, scaled_engine,
                     solve, style)
import inputs

# (MissionCase field, label). Each is scaled by 0.8 and 1.2.
CONSTANTS = [
    ("k_fric", "Skin friction coefficient"),
    ("E_WD", "Fuselage wave drag factor"),
    ("k_pod", "Inlet / pod drag area"),
    ("k_fin", "Fin size"),
    ("C_pb", "Base pressure coefficient"),
    ("f_int", "Interference drag fraction"),
    ("K_ind", "Lift-dependent drag factor"),
    ("K_ram", "Ram factor on thrust lapse"),
    ("T_slope", "Engine thrust per lb"),
    ("tsfc_int", "Engine TSFC"),
    ("d_coef", "Min fuselage diameter (engine)"),
    ("d_clear", "Engine clearance"),
    ("accel_margin", "Thrust margin"),
    ("V_stall", "Stall speed limit"),
    ("C_Lmax", "Max lift coefficient"),
    ("AR", "Wing aspect ratio"),
    ("tau_min", "Min wing t/c"),
    ("wing_weight_factor", "Wing weight factor"),
    ("fuselage_weight_factor", "Fuselage weight factor"),
    ("w_fus", "Fuselage shell weight per area"),
    ("f_other", "Other empty weight fraction"),
    ("W_fixed", "Fixed avionics weight"),
    ("f_res", "Reserve fuel fraction"),
    ("f_idle", "Idle fuel fraction"),
    ("V_climb", "Average climb speed"),
    ("t_super", "Supersonic segment time"),
]

if __name__ == "__main__":
    h = inputs.ALTITUDE if inputs.ALTITUDE is not None else 30000.0
    engine = inputs.ENGINE if inputs.ENGINE not in (None, "auto") else "P400-PRO-LN"
    # Seed: the pinned catalogue engine, then the scalable copy of it from that solution
    _, _, _, seed, note = solve(("seed", h, baseline(), engine))
    if seed is None:
        raise SystemExit(f"{engine} does not close at {h:,.0f} ft: {note}")
    m0 = baseline(**scaled_engine(engine))
    _, _, _, base, note = solve(("baseline", h, m0, None, seed))
    if base is None:
        raise SystemExit(f"scalable {engine} baseline does not close: {note}")
    W0 = base["W_TO"]

    jobs = []
    for name, _ in CONSTANTS:
        v0 = getattr(m0, name)
        for tag, k in (("low", 0.8), ("high", 1.2)):
            jobs.append((f"{name}|{tag}", h, baseline(**{**scaled_engine(engine), name: v0 * k}), None, base))
    results = {label: r for label, _, _, r, _ in run_all(jobs)}

    rows = []
    for name, label in CONSTANTS:
        lo, hi = results[f"{name}|low"], results[f"{name}|high"]
        rows.append((label, getattr(m0, name), None if lo is None else lo["W_TO"] - W0,
                     None if hi is None else hi["W_TO"] - W0))
    no_design = W_CAP - W0
    size = lambda row: max(abs(d) if d is not None else no_design for d in row[2:])
    rows.sort(key=size, reverse=True)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLOT_DIR / "tornado.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["constant", "baseline_value", "dW_TO_minus20_lb", "dW_TO_plus20_lb"])
        for label, v0, dl, dh in rows:
            w.writerow([label, v0, "" if dl is None else round(dl, 3), "" if dh is None else round(dh, 3)])

    print(f"Nominal W_TO {W0:.2f} lb at {h:,.0f} ft, scalable {engine} (engine {base['W_eng']:.2f} lb, "
          f"T_SL {base['T_SL']:.1f} lbf); change in W_TO for -20 % / +20 % on each constant")
    for label, v0, dl, dh in rows:
        fmt = lambda d: f"{d:+7.1f}" if d is not None else "  no design"
        print(f"  {label:34s} {fmt(dl)}  {fmt(dh)}")

    span = max([abs(d) for row in rows for d in row[2:] if d is not None] + [1.0])
    xmax = 1.08 * span
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    style(ax, "")
    ax.set_title(f"Weight sensitivity at {h:,.0f} ft (nominal W_TO = {W0:.1f} lbf)", color=INK, fontsize=13)
    for i, (label, v0, dl, dh) in enumerate(rows):
        y = len(rows) - 1 - i
        for d, color in ((dl, SERIES[0]), (dh, SERIES[1])):
            if d is None:    # no design under the cap: hatched to the edge of the plot, on the side it went
                other = dh if d is dl else dl
                sign = -1 if (other is not None and other > 0) else 1
                ax.barh(y, sign * xmax, height=0.36, color="none", edgecolor=color, hatch="///", linewidth=1)
                ax.annotate("no design", (sign * xmax, y), xytext=(4 * sign, 0), textcoords="offset points",
                            ha="left" if sign > 0 else "right", va="center", fontsize=9, color=INK_2)
            else:
                ax.barh(y, d, height=0.36, color=color)
                ax.annotate(f"{d:+.1f}", (d, y), xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                            ha="left" if d >= 0 else "right", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlim(-1.18 * xmax, 1.18 * xmax)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Change in closed W_TO (lbf)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="-20% on constant"),
                       Patch(color=SERIES[1], label="+20% on constant")],
              frameon=False, fontsize=11, loc="lower right")
    print(f"Saved {save(fig, 'tornado.png')} and tornado.csv")
