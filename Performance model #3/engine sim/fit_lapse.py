"""Refit the mission thrust-lapse approximation after changing the engine cycle.

Run: python "engine sim/fit_lapse.py"
Fits full-throttle Mach 1 net thrust / SLS net thrust over 0-50,000 geometric ft.
"""
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "py-engine-sim" / "src"))

from enginesim.atmosphere import isa
from enginesim.presets import p1000_like
from enginesim.turbojet import FlightCondition, evaluate_turbojet


def fit_lapse():
    engine = p1000_like()
    reference = evaluate_turbojet(engine, FlightCondition())
    rho_sl = isa(0).density_slug_ft3
    sigma, lapse = [], []
    for i in range(501):
        h = i * 100.0
        sigma.append(isa(h).density_slug_ft3 / rho_sl)
        lapse.append(evaluate_turbojet(engine, FlightCondition(h, 1)).net_thrust_lbf / reference.net_thrust_lbf)
    exponent, intercept = statistics.linear_regression(list(map(math.log, sigma)), list(map(math.log, lapse)))
    coefficient = math.exp(intercept)
    error = max(abs(coefficient * s**exponent / l - 1.0) for s, l in zip(sigma, lapse))
    return coefficient, exponent, error


if __name__ == "__main__":
    coefficient, exponent, error = fit_lapse()
    print(f"K_ram = {coefficient:.7f}; LAPSE_EXP = {exponent:.7f}")
    print(f"Maximum relative fit error = {100 * error:.3f}% (Mach 1, 0-50 kft; cycle fit, not engine validation)")
