# Runs the GP in mission_sim.py at a fixed cruise altitude for a series of
# thrust lapse values (cruise thrust / sea-level static thrust) and plots the
# closed takeoff weight. Lapse is set through K_ram: lapse = K_ram sigma^0.9464510.
#   python sweep_lapse.py            # 20,000 ft
#   python sweep_lapse.py 17500      # another altitude
import re
from dataclasses import replace
import subprocess
import sys
from pathlib import Path
from mission_atmosphere import density_ratio
from enginesim.presets import p1000_like
from enginesim.turbojet import FlightCondition, evaluate_turbojet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
H = float(sys.argv[1]) if len(sys.argv) > 1 else 20000.0      # ft
LAPSES = np.round(np.arange(0.20, 1.01, 0.05), 2)

# Exact ISA shared with mission_sim.py.
sigma = density_ratio(H)
LAPSE_EXP = 0.9464510

W_MAX = 55.0
LAPSE_OLD = 0.5 * sigma ** 0.7                 # the original K_ram = 0.5, sigma^0.7 guess
LAPSE_SIM = 1.2666560 * sigma ** LAPSE_EXP     # corrected cycle fit at Mach 1
engine = p1000_like()
reference_thrust = evaluate_turbojet(engine, FlightCondition()).net_thrust_lbf
INLET_LOW, INLET_HIGH = (
    evaluate_turbojet(replace(engine, inlet_pressure_recovery=recovery), FlightCondition(H, 1)).net_thrust_lbf / reference_thrust
    for recovery in (0.80, 1.0)
)

FIELDS = ["W_TO", "W_eng", "T_SL", "L_fus", "d_fus", "W_fus"]


def run_gp(lapse):
    """Solve the GP with h pinned and lapse set; return {field: value} or None."""
    out = subprocess.run([sys.executable, str(HERE / "mission_sim.py"), str(H), "--lapse", str(lapse)],
                         capture_output=True, text=True, cwd=HERE)
    if out.returncode != 0:
        return None
    res = {}
    for name in FIELDS:
        m = re.search(rf"^\s+{name}\s+:\s+([0-9.eE+-]+)", out.stdout, re.M)
        if not m:
            return None
        res[name] = float(m.group(1))
    return res


if __name__ == "__main__":
    rows = [run_gp(l) for l in LAPSES]
    print(f"Cruise altitude {H:,.0f} ft (sigma {sigma:.4f}); lapse = K_ram sigma^{LAPSE_EXP}")
    print("  lapse   W_TO    W_eng    T_SL   L_fus  d_fus[in]")
    for l, r in zip(LAPSES, rows):
        if r is None:
            print(f"  {l:5.2f}   no closure")
        else:
            flag = "  > W_max" if r["W_TO"] > W_MAX else ""
            print(f"  {l:5.2f}  {r['W_TO']:6.2f}  {r['W_eng']:6.2f}  {r['T_SL']:6.1f}  {r['L_fus']:5.2f}  {r['d_fus'] * 12:6.2f}{flag}")

    ok = [(l, r) for l, r in zip(LAPSES, rows) if r is not None]
    x = np.array([l for l, _ in ok])
    W = np.array([r["W_TO"] for _, r in ok])
    Weng = np.array([r["W_eng"] for _, r in ok])

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvspan(INLET_LOW, INLET_HIGH, color="#2a78d6", alpha=0.10, label="Sim range (inlet recovery 0.80 to 1.0)")
    ax.plot(x, W, "o-", color="#2a78d6", lw=2, ms=5, label="Closed W_TO")
    ax.plot(x, Weng, "s--", color="#eb6834", lw=1.5, ms=4, label="Engine weight")
    ax.axhline(W_MAX, color="#d03030", ls="--", lw=1, label=f"W_max = {W_MAX:g} lb")
    for val, name, col in ((LAPSE_OLD, "old guess", "#52514e"), (LAPSE_SIM, "P1000-like sim", "#1baf7a")):
        ax.axvline(val, color=col, ls=":", lw=1.5)
        ax.text(val, ax.get_ylim()[1], f" {name}\n {val:.2f}", va="top", ha="left", fontsize=8, color=col)
    ax.set_xlabel(f"Thrust lapse at {H:,.0f} ft, Mach 1 (T_cruise / T_sea-level static)")
    ax.set_ylabel("Weight (lbf)")
    ax.set_title("Closed takeoff weight vs thrust lapse", loc="left")
    ax.grid(True, color="#e4e3df")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_dir = HERE / "plots" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "lapse_sweep.png", dpi=150, facecolor="#fcfcfb")
    print(f"Saved {out_dir / 'lapse_sweep.png'}")
