# Wing sweep trade: sweep cuts wing wave drag (~cos^2) but raises wing weight
# (~1/cos, Raymer). For each fixed sweep the rest of the design (t/c, wing
# area, fuselage length, altitude pinned at the ceiling) is re-optimized and the
# weight re-closed. Sweep is limited to a delta of aspect ratio AR.
#
#   python sweep_tradeoff.py          -> sweep_tradeoff.png
import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missionsweep import Design, MissionCase, max_sweep_deg, sigma_floor, size_drone

INK, INK2, GRID, SURFACE, BLUE = "#1f2328", "#59636e", "#e6e6e6", "#fcfcfb", "#2a73d4"


def best_at_sweep(m, sweep):
    sig = sigma_floor(m)

    def obj(x):
        r = size_drone(m, Design(sig, *x, m.d_min, sweep))
        return r["W_TO"] if r["converged"] else 1e3

    start = min(((obj((t, s, L)), (t, s, L)) for t in (m.tau_min, 0.04) for s in (1.0, 1.5)
                 for L in np.linspace(m.L_min, m.L_max, 5)))[1]
    res = minimize(obj, start, method="L-BFGS-B", options={"ftol": 1e-12},
                   bounds=[(m.tau_min, 0.2), (1.0, 5.0), (m.L_min, m.L_max)])
    return size_drone(m, Design(sig, *res.x, m.d_min, sweep))


if __name__ == "__main__":
    m = MissionCase(h_max=20000.0, W_max=1e3)
    sweeps = np.linspace(0.0, max_sweep_deg(m), 17)
    rows = [best_at_sweep(m, w) for w in sweeps]

    beta = (max(m.M, 1.2) ** 2 - 1.0) ** 0.5
    W = [r["W_TO"] for r in rows]
    wave_wing = [0.5 * r["rho"] * (m.M * r["a"]) ** 2 * 4.0 / beta * r["tau"] ** 2 * r["cos_sw"] ** 2 * r["S"] for r in rows]
    W_wing = [r["W_wing"] for r in rows]
    D = [r["D"] for r in rows]
    i = int(np.argmin(W))

    panels = [("Closed W_TO (lbf)", W), ("Total drag (lbf)", D),
              ("Wing wave drag (lbf)", wave_wing), ("Wing weight (lbf)", W_wing)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE, sharex=True)
    for ax, (label, y) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.plot(sweeps, y, color=BLUE, lw=2)
        ax.plot(sweeps[i], y[i], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        ax.set_ylabel(label, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("Wing leading-edge sweep (deg)", color=INK2)
    axes[0, 0].annotate(f"min W_TO {W[i]:.1f} lbf at {sweeps[i]:.0f}°", (sweeps[i], W[i]),
                        textcoords="offset points", xytext=(10, 12), fontsize=9, color=INK2)
    fig.suptitle(f"Wing sweep trade at 20,000 ft (delta limit {max_sweep_deg(m):.1f}° for AR {m.AR:g})",
                 x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig("sweep_tradeoff.png", dpi=150)

    print(f"{'sweep':>6} {'W_TO':>7} {'drag':>7} {'wing wave':>10} {'W_wing':>7}")
    for w, a, b, c, d in zip(sweeps, W, D, wave_wing, W_wing):
        print(f"{w:6.1f} {a:7.2f} {b:7.1f} {c:10.1f} {d:7.2f}")
    print("Saved sweep_tradeoff.png")
