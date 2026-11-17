# Evaluates the CFD Gaussian-process surrogates (gp_M*_a*.json): CL, CD, CM at one Mach and angle
# of attack over fuselage diameter, fuselage length, wing half-span and trailing-edge sweep.
#
#   Inputs : fus:diameter [ft], fus:length [ft], wing:span = HALF span [ft], wing:tesweep [deg]
#   Wing   : AR 2, taper 0.4 for these runs; coefficients use the wing area, S = (2 s)^2 / AR
#   Kernel : Matern 5/2 with one length scale per input (the "equations" block in the json)
#
#   python aero_surrogate.py gp_M1.05_a0.json 0.7 10 1.2 40
import json
import math
import sys

import numpy as np

AR_RUN = 2.0
TAPER_RUN = 0.4


class Surrogate:
    def __init__(self, path):
        with open(path) as stream:
            data = json.load(stream)
        self.condition = data["condition"]
        self.params = data["params"]
        self.lo = np.array(data["lo"], dtype=float)
        self.hi = np.array(data["hi"], dtype=float)
        self.models = data["models"]

    def in_range(self, raw):
        raw = np.asarray(raw, dtype=float)
        return bool(np.all(raw >= self.lo) and np.all(raw <= self.hi))

    def predict(self, name, raw):
        """Mean and standard deviation of coefficient `name` at raw inputs (in json order).
        The standard deviation is None when the file has no Kinv."""
        m = self.models[name]
        x = (np.asarray(raw, dtype=float) - self.lo) / (self.hi - self.lo)
        X = np.array(m["X"], dtype=float)
        r = np.sqrt((((x - X) / np.array(m["ls"], dtype=float)) ** 2).sum(axis=1))
        k = m["sf2"] * (1 + math.sqrt(5) * r + 5.0 / 3.0 * r ** 2) * np.exp(-math.sqrt(5) * r)
        mean = m["y_mean"] + m["y_std"] * float(np.dot(m["alpha"], k))
        if "Kinv" not in m:
            return mean, None
        var = m["sf2"] - float(k @ np.array(m["Kinv"], dtype=float) @ k)
        return mean, m["y_std"] * math.sqrt(max(var, 0.0))

    def training_inputs(self, name="CD"):
        """The sampled designs in raw units (the CFD cases the surrogate was fitted to)."""
        return self.lo + np.array(self.models[name]["X"], dtype=float) * (self.hi - self.lo)


def planform(half_span, te_sweep_deg, AR=AR_RUN, taper=TAPER_RUN):
    """Trapezoidal wing from half-span and trailing-edge sweep: area, root/tip chord, LE sweep."""
    b = 2.0 * half_span
    S = b * b / AR
    c_root = 2.0 * S / (b * (1.0 + taper))
    c_tip = taper * c_root
    tan_le = math.tan(math.radians(te_sweep_deg)) + (c_root - c_tip) / half_span
    return {"S": S, "b": b, "c_root": c_root, "c_tip": c_tip, "le_sweep_deg": math.degrees(math.atan(tan_le))}


if __name__ == "__main__":
    sur = Surrogate(sys.argv[1])
    raw = [float(v) for v in sys.argv[2:6]]
    print(f"Mach {sur.condition['mach']}, AoA {sur.condition['aoa']} deg; inputs {dict(zip(sur.params, raw))}")
    if not sur.in_range(raw):
        print("WARNING: outside the sampled range, prediction falls back toward the average")
    p = planform(raw[2], raw[3])
    print(f"Wing: S {p['S']:.3f} ft^2, span {p['b']:.3f} ft, root {p['c_root']:.3f} ft, tip {p['c_tip']:.3f} ft, LE sweep {p['le_sweep_deg']:.1f} deg")
    for name in sur.models:
        mean, sd = sur.predict(name, raw)
        print(f"   {name}: {mean:+.5f}" + (f"  +- {sd:.5f}" if sd is not None else ""))
