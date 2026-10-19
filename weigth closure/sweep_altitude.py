# Runs the GP in mission_sim.py at a series of fixed cruise altitudes and
# plots the optimized takeoff weight.
#   python sweep_altitude.py
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ALTITUDES = np.arange(15000, 20001, 500)       # ft (GP atmosphere fit: 15,000-25,000; ceiling h_max = 20,000)


def run_gp(h):
    """Solve the GP with h pinned; return W_TO in lbf, or nan if infeasible."""
    out = subprocess.run([sys.executable, str(HERE / "mission_sim.py"), str(h)],
                         capture_output=True, text=True, cwd=HERE)
    m = re.search(r"^\s+W_TO\s+:\s+([0-9.]+)", out.stdout, re.M)
    return float(m.group(1)) if out.returncode == 0 and m else float("nan")


if __name__ == "__main__":
    W_max = 55.0
    W = np.array([run_gp(h) for h in ALTITUDES])
    for h, w in zip(ALTITUDES, W):
        print(f"{h:7.0f} ft   " + ("no closure" if np.isnan(w) else f"{w:7.3f} lbf"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ALTITUDES, W, marker="o", ms=4, label="Optimized W_TO")
    ax.axhline(W_max, color="red", linestyle="--", label=f"W_max = {W_max:g} lb")
    ax.set_xlabel("Cruise altitude (ft)")
    ax.set_ylabel("Optimized W_TO (lbf)")
    ax.set_title("Level Mach 1 segment: optimized takeoff weight vs altitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "altitude_sweep.png", dpi=150)
    print("Saved altitude_sweep.png")
