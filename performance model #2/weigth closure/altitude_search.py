"""Search altitude outside the fixed-atmosphere mission formulation.

Keeping ISA values fixed per solve avoids fitting a positive monomial to an
altitude range that contains zero or relaxing the density/altitude relationship.
"""
import math
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from mission_atmosphere import MAX_ALTITUDE_FT, TROPOPAUSE_FT


def altitude_grid(step_ft: float):
    if not math.isfinite(step_ft) or step_ft <= 0:
        raise ValueError("altitude step must be finite and positive")
    points = [i * step_ft for i in range(math.ceil(MAX_ALTITUDE_FT / step_ft))]
    return sorted(set(points + [MAX_ALTITUDE_FT, TROPOPAUSE_FT]))


def run_altitude_search(script, arguments, step_ft=2500.0, run=None):
    with tempfile.TemporaryDirectory(prefix="mission-altitudes-") as directory:
        return _run_altitude_search(script,arguments,step_ft,run,Path(directory))


def _run_altitude_search(script, arguments, step_ft, run, directory):
    run = run or subprocess.run
    best, best_weight, best_altitude = None, math.inf, None
    last_error = ""
    state_file = directory / "previous.json"
    print(f"Exact ISA altitude search, 0-50,000 ft; {step_ft:g} ft spacing plus tropopause", flush=True)
    for altitude in altitude_grid(step_ft):
        warm_args = ["--initial-state",str(state_file)] if state_file.exists() else []
        result = run([sys.executable, str(script), str(altitude), *arguments,
                      *warm_args,"--state-output",str(state_file)], capture_output=True,
                     text=True, cwd=Path(script).parent)
        match = re.search(r"^\s+W_TO\s+:\s+([0-9.eE+-]+)", result.stdout, re.M)
        if result.returncode == 0 and match and math.isfinite(float(match.group(1))):
            weight = float(match.group(1))
            local = state_file.exists() and json.loads(state_file.read_text()).get("mode") == "empirical"
            print(f"  {altitude:9.1f} ft: {weight:.4f} lbf" + (" (checked local solution)" if local else ""), flush=True)
            if weight < best_weight:
                best, best_weight, best_altitude = result, weight, altitude
        else:
            last_error = result.stderr
            if "ModuleNotFoundError" in last_error or "ImportError" in last_error:
                print(last_error, file=sys.stderr)
                return 1
            print(f"  {altitude:9.1f} ft: solver failed / no checked solution (not proof of infeasibility)", flush=True)
    if best is None:
        print("No sampled altitude solved successfully. Last solver error:\n" + last_error, file=sys.stderr)
        return 1
    print(f"\nBest sampled altitude: {best_altitude:.1f} ft (grid search; not a continuous global optimum)")
    print(best.stdout)
    return 0
