# ===========
# Description
# ===========
# Weight and drag breakdown shared by both optimizers (mission_sim.py and
# missionsweep.py). Takes plain numbers, so it works with lcsolver output and
# with the sweep's result dict alike.

import math
from types import SimpleNamespace

from aero import drag_terms


def _v(x):
    return float(x.magnitude if hasattr(x, "magnitude") else x)


def print_breakdown(r, c):
    """r: solved design (names as in mission_sim.py); c: constants it was solved with.

    Needs r: W_TO, W_wing, W_eng, W_other, W_fus, W_f_climb, W_f_cruis(e), W_f_res, S, tau,
             C_L, T_SL, T_seg, lapse, rho, a, L_fus, d_fus, c_sw (or cos_sw)
    Needs c: W_pay, W_fixed, f_other, f_TO, f_res, M, C_f, Swet_S, E_WD, K_ind,
             TW_TO, T_int, T_slope, t_super
    """
    r = {k: _v(x) for k, x in r.items() if isinstance(x, (int, float)) or hasattr(x, "magnitude")}
    W = r["W_TO"]
    f_cruise = r.get("W_f_cruise", r.get("W_f_cruis"))
    c_sw = r.get("c_sw", r.get("cos_sw"))

    print("\nWeight breakdown")
    print("----------------")
    rows = [
        (f"Wing (Raymer x {c.get('wing_weight_factor', 1):g})", r["W_wing"]),
        ("Engine (catalogue / fit)",     r["W_eng"]),
        ("Additional propulsion hardware", r.get("W_prop_install", c.get("W_prop_install", 0.0))),
        ("  fixed avionics/datalink",    c["W_fixed"]),
        ("  other, f_other x W_TO",      c["f_other"] * W),
        (f"Fuselage (shell/frames x {c.get('fuselage_weight_factor', 1):g})", r["W_fus"]),
        ("Fin + strakes (w_surf x area)", r["W_surf"]),
        ("Payload",                      c["W_pay"]),
        ("Fuel: takeoff + climb + accel", r["W_f_climb"]),
        ("Fuel: Mach 1 segment",         f_cruise),
        ("Fuel: reserve + descent",      r["W_f_res"]),
    ]
    for name, w in rows:
        print(f"   {name:<30}: {w:8.3f} lbf   {100 * w / W:5.1f} %")
    other_sum = c["W_fixed"] + c["f_other"] * W + r["W_fus"] + r["W_surf"]
    total = r["W_wing"] + r["W_eng"] + r.get("W_prop_install", c.get("W_prop_install", 0.0)) + other_sum + c["W_pay"] + r["W_f_climb"] + f_cruise + r["W_f_res"]
    print(f"   {'Total':<30}: {total:8.3f} lbf   (W_TO {W:.3f})")
    bind = "Mach 1 segment thrust" if r["T_SL"] > c["TW_TO"] * W * (1 + 1e-6) else "takeoff T/W"
    if r.get("T_SL_path", 0.0) >= r["T_SL"] * (1 - 1e-9) and r["T_SL_path"] > c["TW_TO"] * W:
        bind = "acceleration path (Mach %.2f)" % r["M_path"]
    print(f"   Engine sized by {bind}: T_SL {r['T_SL']:.1f} lbf, lapse {r['lapse']:.3f}, "
          f"takeoff T/W floor {c['TW_TO'] * W:.1f} lbf, thrust per engine lb {r['T_SL'] / r['W_eng']:.1f}")

    # Aero (component build-up, aero.py)
    M = c["M"]
    V = M * r["a"]
    q = 0.5 * r["rho"] * V ** 2
    S = r["S"]
    ns = SimpleNamespace(**c)
    terms, extra = drag_terms(ns, S=S, tau=r["tau"], c_sw=c_sw, L=r["L_fus"], d=r["d_fus"],
                              rho=r["rho"], V=V, M=M, C_L=r["C_L"])
    terms = {k: float(v) for k, v in terms.items()}
    CD = sum(terms.values())
    print("\nAero breakdown")
    print("--------------")
    print(f"   M {M:g}, V {V:.0f} ft/s, q {q:.0f} psf, S {S:.3f} ft^2, fuselage {r['L_fus']:.2f} ft x {r['d_fus'] * 12:.1f} in "
          f"(S_wet {math.pi * r['d_fus'] * r['L_fus']:.1f} ft^2), fin {float(extra['S_v']):.2f} ft^2, strake {float(extra['S_str']):.2f} ft^2")
    print(f"   {'Term':<26} {'C_D':>8}  {'Drag lbf':>9}  {'% of D':>7}")
    for name, cd in terms.items():
        print(f"   {name:<26} {cd:8.5f}  {q * S * cd:9.2f}  {100 * cd / CD:6.1f}%")
    print(f"   {'Total':<26} {CD:8.5f}  {q * S * CD:9.2f}")
    if "D" in r:
        print(f"   (solver drag D = {r['D']:.2f} lbf; difference {q * S * CD - r['D']:+.2f} lbf)")
    print(f"   C_L {r['C_L']:.4f}, L/D {r['C_L'] / CD:.3f}, thrust required {r['T_seg']:.1f} lbf "
          f"(T/W at cruise {r['T_seg'] / W:.2f}), power {r['T_seg'] * V / 550:.0f} hp")
    print(f"   wing sweep {math.degrees(math.acos(max(0.0, min(1.0, c_sw)))):.1f} deg, t/c {r['tau']:.4f}")
