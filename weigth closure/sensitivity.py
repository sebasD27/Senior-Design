# Sensitivity of the closed takeoff weight to the model constants, at a fixed
# cruise altitude (default 20,000 ft, the ceiling in missionsweep.py).
#
# For each perturbation the weight closure is re-run and t/c and wing area are
# re-optimized (altitude is pinned), W_max is lifted so the weight is reported
# even when it does not close under 55 lb.
#
#   python sensitivity.py
#
# Outputs: sensitivity_tornado.png  (+/-20 % on every constant)
#          sensitivity_curves.png   (W_TO vs four levers)
#          sensitivity_vstall.png   (W_TO vs stall speed, for several C_Lmax)
# Also prints the value of each lever that would close at W_max on its own.
import dataclasses
import math

import numpy as np
from scipy.optimize import brentq, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missionsweep import Design, MissionCase, max_sweep_deg, size_drone, sigma_floor

H_TARGET = 30000.0           # ft, altitude the sensitivities are taken at
W_MAX = 55.0                 # lbf
SWING = 0.20                 # +/- fractional perturbation for the tornado

BASE = MissionCase(h_max=H_TARGET)

# constant name -> label shown on the chart
PARAMS = {
    "E_WD": "Fuselage wave drag factor",
    "K_ram": "Ram factor on thrust lapse",
    "d_min": "Min fuselage diameter (engine)",
    "f_other": "Other empty weight fraction",
    "C_f": "Skin friction coefficient",
    "T_slope": "Engine thrust per lb (fit slope)",
    "T_int": "Engine thrust fit intercept",
    "tsfc_int": "TSFC fit intercept",
    "K_ind": "Lift-dependent drag factor",
    "K_wing": "Wing weight coefficient",
    "AR": "Wing aspect ratio",
    "W_fixed": "Fixed avionics weight",
    "V_stall": "Stall speed limit",
    "C_Lmax": "Max lift coefficient",
    "eta_climb": "Climb excess thrust fraction",
    "V_climb": "Average climb speed",
    "f_res": "Reserve fuel fraction",
    "t_super": "Supersonic segment time",
    "TW_TO": "Takeoff T/W",
}


def closed_weight(m: MissionCase) -> float:
    """Lightest closed W_TO at the fixed altitude (t/c and wing area optimized)."""
    sig = max(sigma_floor(m), 0.0) if m.h_max == H_TARGET else sigma_floor(m)

    def obj(x):
        r = size_drone(m, Design(sig, *x))
        return r["W_TO"] if r["converged"] else 1e3

    # x = (t/c, wing area factor, fuselage length, fuselage diameter, sweep)
    best = min(((obj((t, s, L, m.d_min, w)), (t, s, L, m.d_min, w))
                for t in (m.tau_min, 0.04, 0.06) for s in (1.0, 1.5, 2.0)
                for L in np.linspace(m.L_min, m.L_max, 5)
                for w in (0.0, 0.5 * max_sweep_deg(m), max_sweep_deg(m))))
    res = minimize(obj, best[1], method="L-BFGS-B",
                   bounds=[(m.tau_min, 0.2), (1.0, 5.0), (m.L_min, m.L_max), (m.d_min, 2.0),
                           (0.0, max_sweep_deg(m))],
                   options={"ftol": 1e-12})
    return min(res.fun, best[0])


def with_value(name: str, value: float) -> MissionCase:
    return dataclasses.replace(BASE, **{name: value}, W_max=1e3)


def w_of(name: str, value: float) -> float:
    return closed_weight(with_value(name, value))


# --- palette: two validated categorical hues (blue / orange), ink for text ---
INK, INK2, GRID, SURFACE = "#1f2328", "#59636e", "#e6e6e6", "#fcfcfb"
C_DOWN, C_UP = "#2a73d4", "#e8693a"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def tornado(W0):
    rows = []
    for name, label in PARAMS.items():
        lo = w_of(name, getattr(BASE, name) * (1 - SWING))
        hi = w_of(name, getattr(BASE, name) * (1 + SWING))
        rows.append((label, lo - W0, hi - W0))
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))          # biggest on top

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.36, color=C_DOWN, label=f"-{SWING:.0%} on constant")
    ax.barh(y, [r[2] for r in rows], height=0.36, color=C_UP, label=f"+{SWING:.0%} on constant")
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    for yi, (_, lo, hi) in zip(y, rows):
        for v in (lo, hi):
            if abs(v) > 0.05:
                ax.text(v + (0.25 if v > 0 else -0.25), yi, f"{v:+.1f}", va="center",
                        ha="left" if v > 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Change in closed W_TO (lbf)")
    ax.set_title(f"Weight sensitivity at {H_TARGET:,.0f} ft (nominal W_TO = {W0:.1f} lbf)", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig("sensitivity_tornado.png", dpi=150)
    plt.close(fig)
    return rows


def curves(W0, levers):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    print("\nValue of each lever (alone) that closes at W_max = %g lbf" % W_MAX)
    for ax, name in zip(axes.flat, levers):
        base = getattr(BASE, name)
        xs = np.linspace(0.3 * base, 1.5 * base, 13)
        ws = [w_of(name, x) for x in xs]
        style(ax)
        ax.plot(xs, ws, color=C_DOWN, lw=2)
        ax.plot(base, W0, "o", ms=8, color=C_DOWN, mec=SURFACE, mew=2)
        ax.annotate(f"nominal {base:g}", (base, W0), textcoords="offset points",
                    xytext=(8, -14), fontsize=8, color=INK2)
        ax.set_ylim(top=min(max(ws), 2.2 * W0) * 1.05)      # keep a steep curve from squashing the rest
        ax.axhline(W_MAX, color=INK2, ls="--", lw=1)
        ax.annotate(f"W_max = {W_MAX:g} lb", xy=(0.99, W_MAX), xycoords=("axes fraction", "data"),
                    xytext=(0, -4), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color=INK2)
        ax.set_xlabel(f"{PARAMS[name]} ({name})")
        ax.set_ylabel("Closed W_TO (lbf)")
        # lever value that closes alone, if one exists in range
        try:
            x_close = brentq(lambda x: w_of(name, x) - W_MAX, xs[0], xs[-1], xtol=1e-6 * base)
            ax.plot(x_close, W_MAX, "o", ms=8, color=C_UP, mec=SURFACE, mew=2)
            ax.annotate(f"closes at {x_close:.3g}", (x_close, W_MAX), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color=INK2)
            print(f"   {name:10s}: {base:g} -> {x_close:.4g}  ({x_close / base - 1:+.0%})")
        except ValueError:
            print(f"   {name:10s}: cannot close alone within {xs[0]:.3g} - {xs[-1]:.3g}")
    fig.suptitle(f"Levers on closed W_TO at {H_TARGET:,.0f} ft", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig("sensitivity_curves.png", dpi=150)
    plt.close(fig)


def vstall_plot(W0, cl_list=(1.0, 1.2, 1.4), colors=(C_DOWN, C_UP, "#1a9e77")):
    """W_TO vs stall-speed requirement, one curve per C_Lmax (wing loading is
    set by C_Lmax * V_stall^2, so a higher C_Lmax shifts the whole curve)."""
    vs = np.linspace(80.0, 200.0, 25)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    style(ax)
    print("\nStall-speed requirement that closes at W_max = %g lbf" % W_MAX)
    for cl, col in zip(cl_list, colors):
        def w(v):
            return closed_weight(dataclasses.replace(BASE, C_Lmax=cl, V_stall=v, W_max=1e3))
        ws = [w(v) for v in vs]
        ax.plot(vs, ws, color=col, lw=2, label=f"C_Lmax = {cl:g}")
        if ws[0] > W_MAX > ws[-1]:
            v_close = brentq(lambda v: w(v) - W_MAX, vs[0], vs[-1], xtol=0.01)
            ax.plot(v_close, W_MAX, "o", ms=8, color=col, mec=SURFACE, mew=2)
            ax.annotate(f"{v_close:.0f} ft/s ({v_close / 1.688:.0f} kt)", (v_close, W_MAX),
                        textcoords="offset points", xytext=(8, -16), fontsize=8, color=INK2)
            print(f"   C_Lmax {cl:g}: V_stall >= {v_close:.1f} ft/s ({v_close / 1.688:.0f} kt)")
        else:
            print(f"   C_Lmax {cl:g}: never closes; W_TO flattens to {ws[-1]:.1f} lbf at V_stall {vs[-1]:.0f} ft/s")
    ax.axvline(BASE.V_stall, color=INK2, lw=1, ls=":")
    ax.annotate(f"current V_stall {BASE.V_stall:g} ft/s", (BASE.V_stall, 0.97), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.axhline(W_MAX, color=INK2, ls="--", lw=1)
    ax.annotate(f"W_max = {W_MAX:g} lb", xy=(0.99, W_MAX), xycoords=("axes fraction", "data"),
                xytext=(0, -4), textcoords="offset points", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_ylim(top=min(ax.get_ylim()[1], 160))
    ax.set_xlabel("Stall speed requirement V_stall (ft/s)")
    ax.set_ylabel("Closed W_TO (lbf)")
    top = ax.secondary_xaxis("top", functions=(lambda v: v / 1.688, lambda k: k * 1.688))
    top.set_xlabel("knots", color=INK2)
    top.tick_params(colors=INK2, length=0)
    ax.set_title(f"Stall-speed sensitivity at {H_TARGET:,.0f} ft", loc="left", pad=28)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig("sensitivity_vstall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    W0 = closed_weight(dataclasses.replace(BASE, W_max=1e3))
    print(f"Nominal closed W_TO at {H_TARGET:,.0f} ft: {W0:.2f} lbf (W_max {W_MAX:g})")
    rows = tornado(W0)
    print("\nTornado (+/-%d %% on each constant), biggest first" % (SWING * 100))
    for label, lo, hi in reversed(rows):
        print(f"   {label:34s} -{SWING:.0%}: {lo:+6.2f}   +{SWING:.0%}: {hi:+6.2f}")
    curves(W0, ["E_WD", "K_ram", "d_min", "f_other"])
    vstall_plot(W0)
    print("\nSaved sensitivity_tornado.png, sensitivity_curves.png and sensitivity_vstall.png")
