"""Optimizer inputs from the ESP geometry (testDrone.csm design parameters).

The GP uses a simple aircraft: a trapezoidal wing with a straight trailing edge, a round fuselage of
diameter d and length L, a fin sized as k_fin L d, base and inlet drag as fractions of the fuselage frontal
area. csm_inputs() turns the CSM parameters into those quantities:

  wing    reference area, span and MAC with the CSM's own midpoint rule (ogee LE, root chord extended to the
          centerline); the LE sweep is that of the straight-TE trapezoid with the same area, span and tip chord
  fin     exposed part, above the fuselage radius
  base    boattail base minus the nozzle exit of the flow-through duct
  inlet   cowl mouth outside the captured streamtube, plus the diverter plate
  volume  ogive + cylinder + boattail volume relative to the Sears-Haack body the volume check assumes

Not represented: the blended wing-body fairing (extra wetted area), the coke-bottle area ruling and the
S-duct. Fuselage wave drag stays E_WD x Sears-Haack; the CSM's area-distribution report (ar:dqSH, ar:Seff)
is the way to calibrate E_WD for this shape.
"""
import math

# Design parameters of testDrone.csm (ft, deg); names as in the CSM with ':' -> '_'
TEST_DRONE = dict(
    fus_radius=0.30, fus_lnose=4.5, fus_lcyl=3.8, fus_ltail=1.5, fus_rtail=0.20,
    wing_span=1.25, wing_xapex=4.7, wing_xte=7.5, wing_ctip=0.30, wing_inset=0.01, wing_tesweep=0.0,
    wing_hug=2.5, wing_curl=2.5, wing_thickr=0.04, wing_thickt=0.04,
    vtail_croot=1.0, vtail_height=0.9, vtail_taper=0.4, vtail_sweep=50.0, vtail_thick=0.05,
    inlet_w=0.16, inlet_h=0.075, inlet_nsup=2.5, inlet_lipt=0.015, inlet_gap=0.095, inlet_tdiv=0.03,
    inlet_efRatio=1.3, inlet_exitRatio=0.8, inlet_through=1,
)


def _tand(deg):
    return math.tan(math.radians(deg))


def wing_reference(p):
    """Reference area (both sides), span, AR and MAC exactly as the CSM computes them."""
    z0 = p["fus_radius"] - p["wing_inset"]
    D = p["wing_xte"] + p["wing_span"] * _tand(p["wing_tesweep"]) - p["wing_ctip"] - p["wing_xapex"]
    A = c2 = 0.0
    n = 20
    dz = z0 / n
    for k in range(1, n + 1):
        z = (k - 0.5) * dz
        c = p["wing_xte"] + z * _tand(p["wing_tesweep"]) - p["wing_xapex"]
        A, c2 = A + c * dz, c2 + c * c * dz
    n = 400
    for k in range(1, n + 1):
        xa, xb = (k - 1) / n, k / n
        za, zb = (z0 + (p["wing_span"] - z0) * (1 - (1 - x ** p["wing_hug"]) ** p["wing_curl"]) for x in (xa, xb))
        z = 0.5 * (za + zb)
        c = p["wing_xte"] + z * _tand(p["wing_tesweep"]) - (p["wing_xapex"] + D * 0.5 * (xa + xb))
        A, c2 = A + c * (zb - za), c2 + c * c * (zb - za)
    S, b = 2 * A, 2 * p["wing_span"]
    return dict(S=S, b=b, AR=b * b / S, MAC=c2 / A)


def csm_inputs(p=TEST_DRONE):
    """MissionCase fields that pin the GP to this geometry (L, d, S, t/c and sweep fixed)."""
    wing = wing_reference(p)
    R = p["fus_radius"]
    L = p["fus_lnose"] + p["fus_lcyl"] + p["fus_ltail"]
    d = 2 * R
    A_fus = math.pi * R * R
    # straight-TE trapezoid with the same area, span and tip chord -> equivalent LE sweep
    c_root = 2 * wing["S"] / wing["b"] - p["wing_ctip"]
    sweep = math.degrees(math.atan((c_root - p["wing_ctip"]) / p["wing_span"]))
    tau = 0.5 * (p["wing_thickr"] + p["wing_thickt"])
    # exposed fin, above the fuselage radius (root chord sits on the axis)
    ctip_v = p["vtail_croot"] * p["vtail_taper"]
    c_R = p["vtail_croot"] - (p["vtail_croot"] - ctip_v) * R / p["vtail_height"]
    h_exp = p["vtail_height"] - R
    S_fin = h_exp * (c_R + ctip_v) / 2
    # open base: boattail base minus the nozzle exit of the flow-through duct
    ref = math.sqrt(p["inlet_efRatio"] * p["inlet_w"] * p["inlet_h"])
    rex = min(ref * math.sqrt(p["inlet_exitRatio"]), p["fus_rtail"] - 0.03) if p["inlet_through"] else 0.0
    A_open = math.pi * p["fus_rtail"] ** 2 - math.pi * rex ** 2
    # inlet: cowl mouth outside the captured streamtube, plus the diverter
    sup = 1 + 0.152 * (p["inlet_nsup"] - 2)
    A_out = math.pi * (p["inlet_w"] + p["inlet_lipt"]) * (p["inlet_h"] + p["inlet_lipt"]) * sup
    A_cap = math.pi * p["inlet_w"] * p["inlet_h"] * sup
    A_pod = A_out - A_cap + p["inlet_tdiv"] * p["inlet_gap"]
    # body volume (ogive + cylinder + boattail) / Sears-Haack body with the same L and d
    rho = (R * R + p["fus_lnose"] ** 2) / (2 * R)
    n = 2000
    V_nose = sum(math.pi * (R - rho + math.sqrt(rho ** 2 - ((i + 0.5) / n * p["fus_lnose"] - p["fus_lnose"]) ** 2)) ** 2
                 * p["fus_lnose"] / n for i in range(n))
    V = V_nose + A_fus * p["fus_lcyl"] + math.pi * p["fus_ltail"] / 3 * (R * R + R * p["fus_rtail"] + p["fus_rtail"] ** 2)
    volume_ratio = V / (3 * math.pi / 16 * A_fus * L)
    return dict(
        L_min=L, L_max=L, d_min=d, d_max=d,
        S_min=wing["S"], S_max=wing["S"], AR=wing["AR"],
        tau_min=tau, tau_max=tau, sweep_min_deg=sweep, sweep_max_deg=sweep,
        f_strake=0.0,                       # the ogee's inboard LE is already in the reference area
        k_fin=S_fin / (L * d), AR_fin=h_exp ** 2 / S_fin, tau_fin=p["vtail_thick"], sweep_fin_deg=p["vtail_sweep"],
        base_open=A_open / A_fus,
        k_pod=A_pod / A_fus,
        volume_ratio=volume_ratio,          # not a MissionCase field: multiplies f_vol_use in inputs.py
    )


if __name__ == "__main__":
    for key, value in csm_inputs().items():
        print(f"{key:14s} {value:.6g}")
    print({k: round(v, 4) for k, v in wing_reference(TEST_DRONE).items()})
