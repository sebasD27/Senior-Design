# Lowest dash altitude that closes under W_max, with the inputs.py settings.
# The free-altitude GP minimizes weight, which drives the dash high; this instead solves the exact
# fixed-altitude GP on an altitude grid (in parallel) and reports the lowest one that closes.
#   python studies/lowest_altitude.py                 # 0 - 36,000 ft every 2,000 ft
#   python studies/lowest_altitude.py 10000 30000 1000
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # core modules live one folder up
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import inputs
from mission_sim import build_mission, solve_mission


def closed_weight(h):
    """Exact fixed-altitude solve; (W_TO, state) or (None, reason)."""
    try:
        _, _, r = solve_mission(build_mission(h, inputs.mission_case()))
        return h, r["W_TO"], r
    except (RuntimeError, ValueError) as error:
        return h, None, str(error).split("\n")[0][:90]


if __name__ == "__main__":
    lo, hi, step = (float(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (0.0, 36000.0, 2000.0)
    heights = list(np.arange(lo, hi + 1, step))
    with ProcessPoolExecutor(max_workers=min(len(heights), 8)) as pool:
        results = sorted(pool.map(closed_weight, heights))
    W_max = inputs.mission_case().W_max
    print(f"  altitude    W_TO    (cap {W_max:g} lb)")
    for h, W, r in results:
        print(f"  {h:8,.0f}  " + (f"{W:7.2f}   S {r['S']:.2f} ft^2, L {r['L_fus']:.1f} ft, d {r['d_fus']*12:.1f} in, "
                                  f"W_eng {r['W_eng']:.1f}, D {r['D']:.1f} lbf" if W is not None else f"  -      {r}"))
    closed = [h for h, W, _ in results if W is not None and W <= W_max * (1 + 1e-9)]
    print(f"Lowest closing altitude: {min(closed):,.0f} ft" if closed else "No altitude in the grid closes.")
