"""Write testDrone_optimal.csm: the testDrone ESP model resized to an optimizer result.

The optimizer's aircraft is a round fuselage (L, d), a straight-trailing-edge trapezoidal wing (S, AR,
LE sweep, t/c), and a fin, base and inlet set by fractions (k_fin, base_open, k_pod). This keeps the
testDrone design intent (ogee wing, blended wing-body, chin inlet, boattail) and changes only DESPMTR
values so that, measured the same way csm_geometry.csm_inputs() measures the CAD:
  wing reference area, aspect ratio, equivalent LE sweep and t/c match the optimum exactly
  fuselage length and diameter match; the nose / cylinder / boattail split keeps the blended region
  on the cylinder
  exposed fin area (k_fin L d) and aspect ratio match; open base area (base_open x fuselage frontal) matches
  inlet, blend and area-ruling lengths scale with the fuselage (radial sizes with d, axial with L)
The inlet capture area scales with d^2, not with the optimum engine's airflow: resize inlet:w / inlet:h
for the chosen engine.

python make_optimal_csm.py
"""
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "weigth closure"))
from csm_geometry import TEST_DRONE, csm_inputs, wing_reference

# Optimizer result to build (exact fixed-altitude re-solve, see the summary in the chat / README):
# inputs.py with M 1.0, k_fric 0.613, tau_min 0.025, V_stall 120, C_Lmax 1.5, L_min 6, V_climb 300,
# accel_margin 0, engine fits. W_TO 19.60 lb, dash at sea level (the h_min bound).
OPTIMUM = dict(S=0.7635, AR=2.0, c_sw=0.6207, tau=0.025, L=6.0, d=0.3652,
               k_fin=0.25, AR_fin=1.2, tau_fin=0.04, sweep_fin_deg=50.0, base_open=0.5)

# Baseline CSM values that are not in csm_geometry.TEST_DRONE but get rescaled
BASE_EXTRA = {"blend:wmax": 0.70, "blend:xstart": 0.1, "blend:lback": 0.8, "ar:xw": 6.6, "ar:lw": 1.2,
              "ar:wmin": 0.22, "ar:wmineng": 0.27, "ar:engfwd": 0.2, "inlet:xoff": 0.4,
              "inlet:lstraight": 0.4, "inlet:xface": 7.3, "vtail:xroot": 7.6}


def solve(f, lo, hi, n=200):
    """Bisection for f(x) = 0 on [lo, hi] (f increasing)."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def optimal_parameters(opt=OPTIMUM, base=TEST_DRONE):
    p = {}
    R0, L0 = base["fus_radius"], base["fus_lnose"] + base["fus_lcyl"] + base["fus_ltail"]
    R, L = opt["d"] / 2, opt["L"]
    sr, sl = R / R0, L / L0                        # radial and axial scale factors
    # ---- wing: span from S and AR, tip chord from the equivalent sweep, root chord from the ogee area ----
    span = math.sqrt(opt["AR"] * opt["S"]) / 2
    tan_le = math.tan(math.acos(opt["c_sw"]))
    ctip = (2 * opt["S"] / (2 * span) - span * tan_le) / 2
    xte = base["wing_xte"] * sl
    geom = dict(base, fus_radius=R, wing_span=span, wing_ctip=ctip, wing_xte=xte)

    def area_error(c_root):
        return wing_reference(dict(geom, wing_xapex=xte - c_root))["S"] - opt["S"]
    c_root = solve(area_error, ctip + 1e-6, 10 * span)
    xapex = xte - c_root
    # ---- fuselage: nose / cylinder / boattail, blended region on the cylinder ----
    xstart, lback = BASE_EXTRA["blend:xstart"] * sl, BASE_EXTRA["blend:lback"] * sl
    lnose = min(base["fus_lnose"] * sl, xapex - xstart - 0.02)
    ltail = min(base["fus_ltail"] * sl, L - (xte + lback + 0.02))   # boattail starts behind the blend
    if ltail < 0.3 * base["fus_ltail"] * sl:
        raise ValueError("blended region does not fit on the cylinder; shorten blend:lback")
    lcyl = L - lnose - ltail
    # ---- inlet (radial sizes with d, axial with L) and base ----
    inlet = {k: base[k] * sr for k in ("inlet_w", "inlet_h", "inlet_lipt", "inlet_gap", "inlet_tdiv")}
    ref = math.sqrt(base["inlet_efRatio"] * inlet["inlet_w"] * inlet["inlet_h"])
    rex = ref * math.sqrt(base["inlet_exitRatio"])
    rtail = math.sqrt(opt["base_open"] * R * R + rex * rex)   # open base = base_open x pi R^2
    if not rex <= rtail - 0.03 < R:
        raise ValueError("base / nozzle sizes inconsistent")
    # ---- fin: exposed area k_fin L d and exposed AR, overall taper kept ----
    S_fin, taper = opt["k_fin"] * L * opt["d"], base["vtail_taper"]
    h_exp = math.sqrt(opt["AR_fin"] * S_fin)
    height = h_exp + R
    croot = S_fin / (h_exp * ((1 - (1 - taper) * R / height) + taper) / 2)
    xroot = xte + (BASE_EXTRA["vtail:xroot"] - base["wing_xte"]) * sl
    p.update({
        "fus:radius": R, "fus:lnose": lnose, "fus:lcyl": lcyl, "fus:ltail": ltail, "fus:rtail": rtail,
        "wing:span": span, "wing:xapex": xapex, "wing:xte": xte, "wing:ctip": ctip,
        "wing:thickr": opt["tau"], "wing:thickt": opt["tau"],
        "vtail:xroot": xroot, "vtail:croot": croot, "vtail:height": height, "vtail:taper": taper,
        "vtail:sweep": opt["sweep_fin_deg"], "vtail:thick": opt["tau_fin"],
        "blend:wmax": BASE_EXTRA["blend:wmax"] * sr, "blend:xstart": xstart, "blend:lback": lback,
        "ar:xw": BASE_EXTRA["ar:xw"] * sl, "ar:lw": BASE_EXTRA["ar:lw"] * sl, "ar:engfwd": BASE_EXTRA["ar:engfwd"] * sl,
        "ar:wmin": BASE_EXTRA["ar:wmin"] * sr, "ar:wmineng": BASE_EXTRA["ar:wmineng"] * sr,
        "inlet:w": inlet["inlet_w"], "inlet:h": inlet["inlet_h"], "inlet:lipt": inlet["inlet_lipt"],
        "inlet:gap": inlet["inlet_gap"], "inlet:tdiv": inlet["inlet_tdiv"],
        "inlet:xoff": BASE_EXTRA["inlet:xoff"] * sl, "inlet:lstraight": BASE_EXTRA["inlet:lstraight"] * sl,
        "inlet:xface": lnose + (BASE_EXTRA["inlet:xface"] - base["fus_lnose"]) * sl,
    })
    check_asserts(p, base)
    return p


def check_asserts(p, base):
    """The CSM's ASSERT statements, evaluated in Python before writing the file."""
    g = lambda k: p.get(k, base.get(k.replace(":", "_")))
    R, z0 = g("fus:radius"), g("fus:radius") - base["wing_inset"]
    xcyl1, xcyl2 = g("fus:lnose"), g("fus:lnose") + g("fus:lcyl")
    xs = xcyl1 + g("inlet:xoff") + g("inlet:lstraight")
    ref = math.sqrt(base["inlet_efRatio"] * g("inlet:w") * g("inlet:h"))
    rex = min(ref * math.sqrt(base["inlet_exitRatio"]), g("fus:rtail") - 0.03)
    checks = {
        "boattail closes down": R > g("fus:rtail"),
        "finite tip chord": g("wing:ctip") > 0,
        "tip LE behind the apex": g("wing:xte") - g("wing:ctip") > g("wing:xapex"),
        "apex on the cylinder": g("wing:xapex") >= xcyl1 - 1e-6,
        "wing sticks out of the fuselage": g("wing:span") - R > 0.1,
        "root section buried": math.sqrt(R * R - z0 * z0) > 0.5 * g("wing:thickr") * (g("wing:xte") - g("wing:xapex")),
        "blend starts on the cylinder": g("wing:xapex") - g("blend:xstart") >= xcyl1 - 1e-6,
        "blend ends on the cylinder": xcyl2 >= g("wing:xte") + g("blend:lback") - 1e-6,
        "blend wider than the cylinder": g("blend:wmax") - R > 0.05,
        "blend narrower than the wing": g("wing:span") - g("blend:wmax") > 0.1,
        "engine floor >= forward floor": g("ar:wmineng") >= g("ar:wmin"),
        "inlet lip behind the seam": g("inlet:xoff") > 0.05,
        "engine face behind the straight run": g("inlet:xface") > xs,
        "engine face on the cylinder": xcyl2 > g("inlet:xface"),
        "duct fits inside the flare": 0.85 * R - ref - g("inlet:lipt") > 0,
        "mouth narrower than the fuselage": R - g("inlet:w") - g("inlet:lipt") > 0,
        "nozzle stays open": rex > 0.02,
        "fin inside the fuselage length": g("vtail:xroot") + g("vtail:croot") <= xcyl2 + g("fus:ltail"),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError("CSM asserts would fail: " + ", ".join(failed))


def write_csm(p, src=HERE / "testDrone.csm", dst=HERE / "testDrone_optimal.csm"):
    text = src.read_text()
    for name, value in p.items():
        pattern = rf"^(DESPMTR\s+{re.escape(name)}\s+)(\S+)"
        text, n = re.subn(pattern, lambda m: m.group(1) + f"{value:.6g}".ljust(len(m.group(2))), text, flags=re.M)
        if n != 1:
            raise ValueError(f"{name}: found {n} DESPMTR lines")
    header = ("# testDrone_optimal: testDrone resized to the optimizer result by make_optimal_csm.py\n"
              "# (only DESPMTR values differ from testDrone.csm; the P400 comments refer to the baseline)\n")
    dst.write_text(header + text)
    return dst


if __name__ == "__main__":
    params = optimal_parameters()
    out = write_csm(params)
    for k, v in params.items():
        print(f"   {k:16s} {v:.4f}")
    # check: measure the new CSM the way the optimizer reads the CAD
    measured = csm_inputs({**TEST_DRONE, **{k.replace(":", "_"): v for k, v in params.items()
                                            if k.replace(":", "_") in TEST_DRONE}})
    print(f"Wrote {out}")
    print(f"Check: S {measured['S_min']:.4f} (target {OPTIMUM['S']}), AR {measured['AR']:.4f} ({OPTIMUM['AR']}), "
          f"LE sweep {measured['sweep_min_deg']:.2f} deg ({math.degrees(math.acos(OPTIMUM['c_sw'])):.2f}), "
          f"t/c {measured['tau_min']}, L {measured['L_min']:.3f}, d {measured['d_min']:.4f}, "
          f"k_fin {measured['k_fin']:.4f} ({OPTIMUM['k_fin']}), AR_fin {measured['AR_fin']:.3f}, "
          f"base_open {measured['base_open']:.4f} ({OPTIMUM['base_open']}), k_pod {measured['k_pod']:.4f}")
