"""Thrust lapse contour for the P1000-like turbojet over Mach and altitude.

lapse = net thrust(M, h) / sea-level static net thrust (M 0, h 0), full throttle.
Run from anywhere:  python "engine sim/lapse_contour.py"
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "py-engine-sim" / "src"))

from enginesim.presets import p1000_like
from enginesim.atmosphere import isa
from enginesim.turbojet import FlightCondition, evaluate_turbojet

engine = p1000_like()
mach = np.linspace(0.0, 1.5, 61)
alt = np.linspace(0.0, 50000.0, 101)

T_ref = evaluate_turbojet(engine, FlightCondition(altitude_ft=0.0, mach=0.0), throttle=1.0).net_thrust_lbf
thrust = np.zeros((len(alt), len(mach)))
sfc = np.zeros_like(thrust)
for i, h in enumerate(alt):
    for j, m in enumerate(mach):
        r = evaluate_turbojet(engine, FlightCondition(altitude_ft=h, mach=m), throttle=1.0)
        thrust[i, j] = r.net_thrust_lbf
        sfc[i, j] = r.specific_fuel_consumption_lbm_lbf_h
lapse = thrust / T_ref

(HERE / "plots").mkdir(parents=True, exist_ok=True)
np.savetxt(HERE / "plots" / "lapse_grid.csv", lapse, delimiter=",",
           header="rows: geometric altitude 0-50000 ft (101), cols: Mach 0-1.5 (61); lapse = T / T_SLS")

def contour(Z, levels, title, cbar, fname, cmap, fmt="%.2f"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cf = ax.contourf(mach, alt / 1000, Z, levels=levels, cmap=cmap)
    cs = ax.contour(mach, alt / 1000, Z, levels=levels, colors="k", linewidths=0.5)
    ax.clabel(cs, fmt=fmt, fontsize=8)
    fig.colorbar(cf, label=cbar)
    ax.plot(1.0, 20.0, "r*", ms=14, label="Mach 1, 20,000 ft (design point)")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Altitude (kft)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / fname, dpi=150)
    plt.close(fig)

contour(lapse, 14, f"P1000-like thrust lapse, T / T_SLS (T_SLS = {T_ref:.0f} lbf)",
        "Thrust lapse", "lapse_contour.png", "viridis")
contour(sfc, 14, "P1000-like TSFC (lbm/lbf/h), full throttle",
        "TSFC", "tsfc_contour.png", "magma")

j1 = np.argmin(abs(mach - 1.0)); i20 = np.argmin(abs(alt - 20000))
print(f"T_SLS = {T_ref:.1f} lbf")
sigma20 = isa(alt[i20]).density_slug_ft3 / isa(0).density_slug_ft3
print(f"lapse at M 1.0, 20,000 ft = {lapse[i20, j1]:.3f}  (0-50 kft screening fit = {1.2666560 * sigma20**0.9464510:.3f})")
print(f"lapse range: {lapse.min():.3f} - {lapse.max():.3f}")
