# Drag vs velocity for the optimized aircraft at its design altitude:
# total, parasitic (friction + wave, zero-lift) and induced (lift-dependent),
# plus the thrust available. Same drag model as missionsweep.py.
#   python drag_vs_speed.py          -> drag_vs_speed.png
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missionsweep import MissionCase, size_drone, sweep_designs

INK, INK2, GRID, SURFACE = "#1f2328", "#59636e", "#e6e6e6", "#fcfcfb"
C_TOTAL, C_PARA, C_IND = "#2a73d4", "#e8693a", "#1a9e77"


def drag_parts(V, res, m: MissionCase):
    """Parasitic and induced drag (lbf) at speed V (ft/s), weight and altitude of the design."""
    V = np.asarray(V, dtype=float)
    M = V / res["a"]
    beta = np.sqrt(np.maximum(M, 1.2) ** 2 - 1.0)       # held at M 1.2 values below 1.2
    q = 0.5 * res["rho"] * V ** 2
    S, tau = res["S"], res["tau"]
    C_L = res["W_cr_start"] / (q * S)
    parasitic = q * ((m.C_f * m.Swet_S * S + m.C_f * res["Swet_fus"])
                     + 4.0 / beta * tau ** 2 * res["cos_sw"] ** 2 * S + res["CDA_fus_w"])
    induced = q * S * m.K_ind * beta / 4.0 * C_L ** 2
    return parasitic, induced


if __name__ == "__main__":
    m = MissionCase(h_max=20000.0)
    design, _ = sweep_designs(m)
    res = size_drone(m, design)
    a = res["a"]

    V = np.linspace(0.3 * a, 2.0 * a, 400)
    para, ind = drag_parts(V, res, m)
    total = para + ind
    T_avail = res["lapse"] * res["T_SL"]                # turbojet: ~constant with speed
    V_cr = m.M * a
    p_cr, i_cr = drag_parts(V_cr, res, m)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)

    ax.axvspan(V[0], 1.2 * a, color=INK2, alpha=0.08, lw=0)
    ax.annotate("M < 1.2: drag coefficients\nheld at M 1.2 values", (V[0], 0.97),
                xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points",
                va="top", fontsize=8, color=INK2)
    ax.plot(V, total, color=C_TOTAL, lw=2.2, label="Total drag")
    ax.plot(V, para, color=C_PARA, lw=2, label="Parasitic (friction + wave)")
    ax.plot(V, ind, color=C_IND, lw=2, label="Induced (lift-dependent)")
    ax.axhline(T_avail, color=INK2, ls="--", lw=1.2)
    ax.annotate(f"Thrust available {T_avail:.0f} lbf", xy=(0.99, T_avail), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.plot(V_cr, p_cr + i_cr, "o", ms=8, color=C_TOTAL, mec=SURFACE, mew=2)
    ax.annotate(f"Design point, Mach {m.M:g}: {p_cr + i_cr:.0f} lbf", (V_cr, p_cr + i_cr),
                textcoords="offset points", xytext=(10, 12), fontsize=9, color=INK2)

    top = ax.secondary_xaxis("top", functions=(lambda v: v / a, lambda M: M * a))
    top.set_xlabel("Mach", color=INK2)
    top.tick_params(colors=INK2, length=0)
    ax.set_xlabel("Velocity (ft/s)", color=INK2)
    ax.set_ylabel("Force (lbf)", color=INK2)
    ax.set_ylim(0, min(total.max(), 4 * T_avail) * 1.05)
    ax.set_title(f"Drag vs velocity at {res['h']:,.0f} ft, W = {res['W_cr_start']:.1f} lbf",
                 loc="left", color=INK, pad=28)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig("drag_vs_speed.png", dpi=150)
    plt.close(fig)

    print(f"{'Mach':>5} {'V [ft/s]':>9} {'total':>7} {'parasitic':>10} {'induced':>8}")
    for M in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
        p, i = drag_parts(M * a, res, m)
        print(f"{M:5.1f} {M * a:9.0f} {p + i:7.1f} {p:10.1f} {i:8.1f}")
    print(f"Thrust available {T_avail:.1f} lbf. Saved drag_vs_speed.png")
