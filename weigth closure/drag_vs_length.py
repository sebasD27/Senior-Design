# Sweeps fuselage length and plots total, friction and wave drag on one plot.
#
# The aircraft is the optimum from missionsweep.py at the target altitude
# (wing area, t/c, weight, altitude held fixed); only the fuselage length
# changes, with the diameter at its engine minimum. So this is the pure
# aerodynamic trade: a longer fuselage cuts fuselage wave drag (~1/L^2) but adds
# wetted area and skin friction (~L). Weight is NOT re-closed here, so the drag
# minimum can sit slightly away from the weight-optimal length.
#
#   python drag_vs_length.py          -> drag_vs_length.png
import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missionsweep import MissionCase, size_drone, sweep_designs

INK, INK2, GRID, SURFACE = "#1f2328", "#59636e", "#e6e6e6", "#fcfcfb"
C_TOTAL, C_FRIC, C_WAVE = "#2a73d4", "#e8693a", "#1a9e77"


def drag_components(L, d, res, m: MissionCase):
    """Friction, wave and lift-dependent drag (lbf) for fuselage length L and diameter d."""
    S, tau = res["S"], res["tau"]
    q = 0.5 * res["rho"] * (m.M * res["a"]) ** 2
    beta = math.sqrt(max(m.M, 1.2) ** 2 - 1.0)
    Swet_fus = math.pi * d * L
    CDA_fus = m.E_WD * 4.5 * math.pi * (math.pi * d ** 2 / 4.0 / L) ** 2
    fric = q * (m.C_f * m.Swet_S * S + m.C_f * Swet_fus)
    wave = q * (4.0 / beta * tau ** 2 * res["cos_sw"] ** 2 * S + CDA_fus)
    lift = q * S * m.K_ind * beta / 4.0 * res["C_L"] ** 2
    return fric, wave, lift


if __name__ == "__main__":
    m = MissionCase(h_max=20000.0)
    design, _ = sweep_designs(m)
    res = size_drone(m, design)
    d = design.d_fus

    L = np.linspace(m.L_min, m.L_max, 200)
    fric, wave, lift = np.array([drag_components(l, d, res, m) for l in L]).T
    total = fric + wave + lift
    i = int(np.argmin(total))

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)

    ax.plot(L, total, color=C_TOTAL, lw=2.2, label="Total drag")
    ax.plot(L, fric, color=C_FRIC, lw=2, label="Friction drag (wing + fuselage)")
    ax.plot(L, wave, color=C_WAVE, lw=2, label="Wave drag (wing + fuselage)")
    ax.plot(L[i], total[i], "o", ms=8, color=C_TOTAL, mec=SURFACE, mew=2)
    ax.annotate(f"min drag {total[i]:.1f} lbf at L = {L[i]:.1f} ft", (L[i], total[i]),
                textcoords="offset points", xytext=(10, 14), fontsize=9, color=INK2)
    ax.axvline(design.L_fus, color=INK2, lw=1, ls=":")
    ax.annotate(f"weight-optimal L = {design.L_fus:.1f} ft", (design.L_fus, 0.02),
                xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=INK2)

    ax.set_xlabel(f"Fuselage length (ft), diameter {d * 12:.0f} in")
    ax.set_ylabel("Drag (lbf)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Drag vs fuselage length at {res['h']:,.0f} ft, Mach {m.M:g}", loc="left", color=INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig("drag_vs_length.png", dpi=150)
    plt.close(fig)

    print(f"Fixed aircraft: S = {res['S']:.2f} ft^2, t/c = {res['tau']:.3f}, W = {res['W_cr_start']:.1f} lbf")
    print(f"{'L [ft]':>8} {'total':>8} {'friction':>9} {'wave':>8}   [lbf]")
    for l in np.linspace(m.L_min, m.L_max, 9):
        f_, w_, li = drag_components(l, d, res, m)
        print(f"{l:8.1f} {f_ + w_ + li:8.1f} {f_:9.1f} {w_:8.1f}")
    print("Saved drag_vs_length.png")
