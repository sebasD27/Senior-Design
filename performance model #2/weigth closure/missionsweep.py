# ===========
# Description
# ===========
# Traditional weight closure + design sweep for the supersonic drone.
# Same models and constants as mission_sim.py (the GP version), solved the
# classic way:
#
#   1. Inner loop (weight closure): for a fixed design (altitude, t/c, wing
#      size) guess W_TO, fly the mission, rebuild W_TO from the pieces, and
#      iterate until the guess and the build-up agree.
#   2. Outer loop (design sweep): repeat the closure over a grid of designs
#      and keep the lightest one that closes under W_max. A bounded local
#      optimizer then polishes the best grid point.
#
# The GP turns every "==" of the physics into ">=" and lets the optimizer pick
# which ones are tight. Here the physics is written as equalities, and the
# design freedoms the GP has (altitude, tau, S above the stall minimum, engine
# sized by the worse of takeoff and cruise) become the sweep variables / max().

# =================
# Import Statements
# =================
import math
import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from aero import drag_terms
from engine_choice import add_hardware_arguments, catalogue_engine, lapse_map, mission_lapse
from mission_atmosphere import atmosphere_at_altitude, altitude_from_sigma, density_ratio, RHO_SL
from mission_path import flight_segments, segment_loads, path_fuel, thrust_lapse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class MissionCase:
    engine_model: str | None = None  # fixed catalogue engine; None allows continuous resizing
    W_prop_install: float = 0.0     # lb, additional installation hardware beyond existing allowances
    W_wing_min: float = 0.0         # lb, optional measured/structurally justified floor
    W_fus_min: float = 0.0          # lb, optional measured/structurally justified floor
    wing_weight_factor: float = 2.0       # provisional structural allowance on the Raymer estimate
    fuselage_weight_factor: float = 1.5   # provisional allowance on shell + frames estimate
    lapse_override: float | None = None  # total cruise net thrust / SLS thrust, scenario assumption
    # Mission requirements
    W_max: float = 55.0          # lbf, max takeoff weight (Part 107)
    W_pay: float = 0.0           # lbf, payload
    t_super: float = 5.0         # s, level supersonic flight time
    M: float = 1.0               # cruise Mach (drag held at M 1.2 values below 1.2)
    rho_SL: float = RHO_SL      # slug/ft^3, exact shared ISA reference
    V_stall: float = 120.0       # ft/s
    C_Lmax: float = 1.0
    TW_TO: float = 0.5           # required takeoff T/W
    # Propulsion
    # Engine fits from AMT / JetCat / PBS data (engine weight W in lb):
    #   max static thrust  T_SL  = T_slope * W + T_int           [lbf]
    #   TSFC at max thrust c_T   = tsfc_int + tsfc_slope * W      [1/hr]
    T_slope: float = 7.1
    T_int: float = 28.6          # lbf
    tsfc_int: float = 1.617      # 1/hr
    tsfc_slope: float = -0.0117  # 1/hr per lb
    K_ram: float = 1.2666560     # unvalidated screening-cycle fit, Mach 1, 0-50 kft
    lapse_exp: float = 0.9464510 # fitted by engine sim/fit_lapse.py
    # Aerodynamics
    # Skin friction per component, C_f = cf_A Re^-cf_n (turbulent flat plate, fit to Raymer's
    # 0.455/log10(Re)^2.58 over Re 1e6-5e7, +-1.3%), see aero.py. mu: air viscosity at 20,000 ft.
    cf_A: float = 0.04254
    cf_n: float = 0.1641
    mu: float = 3.32e-7          # slug/(ft s)
    Swet_S: float = 2.05         # wetted / planform area of a thin lifting surface
    # Calibration: total friction at the hand-estimate geometry (10 ft x 6.8 in body, 2 ft^2 wing, 50 lb,
    # Mach 1 at 20,000 ft) was 0.0734 ft^2 here against 0.045 ft^2 in the hand estimate; 0.045/0.0734.
    # Set to 1.0 for the uncalibrated Raymer build-up.
    k_fric: float = 0.613
    # Vertical fin (elevons, so no horizontal tail), sized as a fraction of the fuselage side area
    # L d; strakes ride on the wing area. PLACEHOLDERS: set from the CAD.
    k_fin: float = 0.25          # fin area / (fuselage length x diameter)
    AR_fin: float = 1.2
    tau_fin: float = 0.04
    sweep_fin_deg: float = 50.0
    Q_fin: float = 1.05          # interference factor on the fin
    f_strake: float = 0.15       # strake area / wing area (extra wetted area, no lift credit)
    AR_str: float = 0.6
    sweep_str_deg: float = 70.0
    w_surf: float = 0.40         # lbf/ft^2 planform, fin and strake structure
    # Base drag, inlet / ventral pod and interference (see aero.py). PLACEHOLDERS: set from the CAD
    C_pb: float = 0.15           # base drag coefficient on the open base area (-C_p at the base, Mach ~1)
    base_open: float = 0.5       # fraction of the fuselage base not filled by the exhaust nozzle
    CD_pod: float = 0.2          # inlet / pod drag coefficient on its frontal area (near the drag-rise peak)
    k_pod: float = 0.3           # pod frontal area / fuselage frontal area
    f_int: float = 0.10          # interference + excrescences, fraction of the other parasitic drag
    # Fuselage: Swet = pi d L, wave drag area = E_WD (9 pi / 2)(A_max / L)^2
    # (Sears-Haack), A_max = pi d^2 / 4. E_WD = 1.8 reproduces the old 6 ft x 8 in
    # body (0.086 ft^2).
    E_WD: float = 1.8
    # Fuselage diameter follows the engine: d = d_coef (W_eng / 1 lb)^d_exp, a power law fitted
    # to the catalogue (propulsion/engine_catalogue.csv, 11 engines, +-6%). Wider than this is only
    # worse (drag), so the optimizer sits on it; d_min is just an absolute floor on the design variable.
    w_fus: float = 0.35          # lbf/ft^2 of fuselage wetted area, CFRP shell + frames (1 mm laminate = 0.32 lb/ft^2)
    W_eng_min: float = 4.0       # lbf, lightest catalogue engine (P220 4.08 lb); the thrust fit is invalid below
    W_eng_max: float = 43.21     # lbf, heaviest engine in engine_catalogue.csv (TJ150)
    d_coef: float = 2.5282 / 12.0    # ft at W_eng = 1 lb
    d_exp: float = 0.3969
    d_min: float = 0.25          # ft, floor on the fuselage-diameter design variable
    L_min: float = 4.0           # ft, fuselage length bounds (placeholders)
    L_max: float = 12.0
    K_ind: float = 1.2
    # Structures / weights
    AR: float = 2.0
    N_ult: float = 6.0
    K_wing: float = 0.0103       # Raymer fighter wing coefficient (1/cos(sweep) is applied separately)
    tau_min: float = 0.025
    f_other: float = 0.20
    W_fixed: float = 3.0         # lbf
    f_TO: float = 0.02
    V_climb: float = 600.0       # ft/s
    eta_climb: float = 0.5      # legacy input; finite-segment fuel no longer assumes this fraction
    # Finite climb and acceleration use the same sampled route as mission_sim.
    # A lapse override is a constant scenario over that route, not a bypass.
    accel_margin: float = 0.10
    M_accel_start: float = 0.6  # legacy input; route now starts at 1.2 V_stall
    g: float = 32.174            # ft/s^2
    # Exact ISA (troposphere + isothermal lower stratosphere), 0-50 kft.
    sig_max: float = 1.0
    sig_min: float = density_ratio(50000.0)
    h_max: float = 50000.0       # ft, configurable cruise ceiling within ISA mission domain
    f_res: float = 0.06


def max_sweep_deg(m: MissionCase) -> float:
    """Most sweep a wing of aspect ratio AR can have as a delta (AR = 4 / tan(sweep))."""
    return math.degrees(math.atan(4.0 / m.AR))


def sigma_floor(m: MissionCase) -> float:
    """Lowest allowed density ratio from the exact ISA altitude ceiling."""
    return max(m.sig_min, density_ratio(m.h_max))


@dataclass
class Design:
    """The free design choices the GP gets to optimize."""
    sigma: float                 # cruise density ratio (sets altitude)
    tau: float                   # wing t/c
    S_factor: float = 1.0        # wing area / stall-limited minimum area
    L_fus: float = 6.0           # ft, fuselage length
    d_fus: float = 8.0 / 12.0    # ft, fuselage diameter
    sweep_deg: float = 60.0      # deg, wing leading-edge sweep


def _evaluate(W_TO: float, mission: MissionCase, design: Design, c_T: float, d_fus: float):
    """Fly the mission once at a guessed W_TO and TSFC. Same names as mission_sim.py."""
    m = mission
    sigma, tau = design.sigma, design.tau

    # Atmosphere
    h = altitude_from_sigma(sigma)
    atmosphere = atmosphere_at_altitude(h)
    rho = atmosphere.density_slug_ft3
    a = atmosphere.speed_of_sound_ft_s
    V = m.M * a
    q = 0.5 * rho * V ** 2
    lapse = thrust_lapse(m, h, m.M)

    beta = math.sqrt(max(m.M, 1.2) ** 2 - 1.0)
    K_sup = beta / 4.0
    K_wave = 4.0 / beta

    # Wing sized by stall speed (S_factor >= 1 allows a bigger wing)
    S = design.S_factor * 2.0 * W_TO / (m.rho_SL * m.V_stall ** 2 * m.C_Lmax)

    # Size and time the same route as mission_sim, using takeoff weight for
    # conservative path loads. A fixed engine is never resized to hide failure.
    cos_sw = math.cos(math.radians(design.sweep_deg))
    loads = segment_loads(m, flight_segments(m, h), weight=W_TO, S=S, tau=tau,
                         c_sw=cos_sw, L=design.L_fus, d=d_fus)
    T_path, M_path = 0.0, m.M
    for segment, drag, low, high in loads:
        need = (1 + m.accel_margin) * drag / low
        if need > T_path:
            T_path = need
            M_path = max(p.mach for p in segment.points)
    engine = catalogue_engine(m.engine_model) if m.engine_model else None
    T_SL = engine.thrust if engine else max(m.TW_TO * W_TO, T_path) * (1 + 1e-8)
    fuel, path_time = path_fuel(m, loads, W_TO, T_SL, c_T)
    W_f_climb = m.f_TO * W_TO + fuel
    fuel_valid = math.isfinite(W_f_climb) and W_f_climb < W_TO
    W_cr_start = max(W_TO - W_f_climb, 0.0)

    # Level-flight aerodynamics (lift = weight at start of the Mach 1 segment)
    C_L = W_cr_start / (q * S)
    Swet_fus = math.pi * d_fus * design.L_fus
    CDA_fus_w = m.E_WD * 4.5 * math.pi * (math.pi * d_fus ** 2 / 4.0 / design.L_fus) ** 2
    cos_sw = math.cos(math.radians(design.sweep_deg))
    # Component drag build-up (aero.py): friction with Re, form factors, fin and strake
    terms, extra = drag_terms(m, S=S, tau=tau, c_sw=cos_sw, L=design.L_fus, d=d_fus,
                              rho=rho, V=V, M=m.M, C_L=C_L)
    C_D0 = sum(v for k, v in terms.items() if k.startswith("Skin"))
    C_Dw = sum(v for k, v in terms.items() if k.startswith("Wave"))   # (base, pod, interference are in C_D only)
    C_D = sum(terms.values())
    S_v = extra["S_v"]
    D = q * C_D * S

    # Power-based segment: level flight, thrust = drag, power = T V
    T_seg = D                                            # thrust required
    P_req = T_seg * V / 550.0                            # hp

    # Engine sized by the worst of takeoff T/W, segment thrust and the acceleration path
    T_SL_required = max(m.TW_TO * W_TO, T_seg / lapse, T_path)
    # The path includes the dash endpoint at takeoff weight, so it also
    # conservatively covers steady dash drag after fuel burn.

    # Empty weight build-up
    W_wing = max(m.W_wing_min, m.wing_weight_factor * m.K_wing * math.sqrt(m.N_ult * W_TO) * S ** 0.622
                 * m.AR ** 0.785 * tau ** -0.4 / cos_sw)
    W_eng = engine.weight if engine else max(max(T_SL - m.T_int, 0.0) / m.T_slope, m.W_eng_min)
    W_other = m.W_fixed + m.f_other * W_TO
    W_fus = max(m.W_fus_min, m.fuselage_weight_factor * m.w_fus * Swet_fus)
    W_surf = m.w_surf * (S_v + extra["S_str"])             # fin + strake structure
    W_empty = W_wing + W_eng + m.W_prop_install + W_other + W_fus + W_surf
    W_ZF = m.W_pay + W_empty

    # Reserve, and segment fuel = TSFC * thrust * time (no Breguet)
    W_f_res = m.f_res * W_ZF
    W_cr_end = W_ZF + W_f_res
    W_f_cruise = (c_T / 3600.0) * T_seg * m.t_super

    return {
        "W_TO": W_TO, "W_ZF": W_ZF, "W_empty": W_empty, "W_wing": W_wing,
        "W_eng": W_eng, "W_prop_install": m.W_prop_install, "T_SL_required": T_SL_required,
        "T_SL_path": T_path, "M_path": M_path,
        "path_time": path_time, "fuel_valid": fuel_valid,
        "W_other": W_other, "W_fus": W_fus, "W_surf": W_surf, "S_v": S_v, "W_f_climb": W_f_climb,
        "W_f_cruise": W_f_cruise, "c_T": c_T, "L_fus": design.L_fus, "d_fus": d_fus,
        "Swet_fus": Swet_fus, "CDA_fus_w": CDA_fus_w, "sweep_deg": design.sweep_deg, "cos_sw": cos_sw, "W_f_res": W_f_res, "W_cr_start": W_cr_start,
        "W_cr_end": W_cr_end, "S": S, "tau": tau, "C_L": C_L, "C_D": C_D,
        "C_D0": C_D0, "C_Dw": C_Dw, "D": D, "T_SL": T_SL, "T_seg": T_seg, "P_req": P_req,
        "h": h, "rho": rho, "a": a, "lapse": lapse,
        # build-up: what the mission says W_TO has to be
        "W_TO_calc": W_ZF + W_f_climb + W_f_cruise + W_f_res,
    }


def evaluate_one(W_TO: float, mission: MissionCase, design: Design):
    """TSFC depends on engine weight, which depends on thrust, which depends on
    the climb fuel (via TSFC): iterate c_T to a fixed point."""
    m = mission
    if not all(math.isfinite(v) and v >= 1 for v in (m.wing_weight_factor,m.fuselage_weight_factor)):
        raise ValueError("structural weight factors must be finite and at least 1")
    if not math.isfinite(m.W_prop_install) or m.W_prop_install < 0 or not math.isfinite(m.W_pay) or m.W_pay < 0:
        raise ValueError("installation and payload weights must be finite and nonnegative")
    if m.engine_model:
        engine = catalogue_engine(m.engine_model)
        r = _evaluate(W_TO, m, design, engine.tsfc, max(design.d_fus, m.d_min, engine.diameter))
        r["propulsion_valid"] = r["fuel_valid"] and r["T_SL_required"] <= engine.thrust * (1 + 1e-7)
        if not r["propulsion_valid"]:
            r["propulsion_error"] = "selected engine cannot supply required thrust"
        return r
    c_T = m.tsfc_int + m.tsfc_slope * (m.W_eng_min + m.W_eng_max) / 2.0
    if not math.isfinite(c_T) or c_T <= 0 or not 0 < m.W_eng_min <= m.W_eng_max:
        raise ValueError("engine-fit bounds must be ordered and give positive TSFC")
    d = max(design.d_fus, m.d_min)
    for _ in range(500):
        r = _evaluate(W_TO, m, design, c_T, d)
        r["propulsion_valid"] = False
        if not r["fuel_valid"]:
            r["propulsion_error"] = "flight path has no finite fuel solution within the thrust margin"
            return r
        if not math.isfinite(r["W_eng"]) or r["W_eng"] > m.W_eng_max:
            r["propulsion_error"] = "required engine exceeds catalogue fit range"
            return r
        c_new = m.tsfc_int + m.tsfc_slope * r["W_eng"]
        if not math.isfinite(c_new) or c_new <= 0:
            r["propulsion_error"] = "engine fit gives nonpositive TSFC"
            return r
        # fuselage must at least hold the engine: diameter fit on engine weight
        d_new = max(design.d_fus, m.d_coef * r["W_eng"] ** m.d_exp)
        if abs(c_new - c_T) < 1e-12 and abs(d_new - d) < 1e-12:
            r["propulsion_valid"] = True
            return r
        c_T, d = c_new, 0.5 * (d + d_new)        # damped: the engine-body-drag loop can overshoot
    r["propulsion_error"] = "engine/diameter iteration did not converge"
    return r


def size_drone(mission: MissionCase, design: Design, W0_guess: float = 50.0,
               relax: float = 0.5, tol: float = 1e-10, max_iter: int = 500):
    """Traditional fixed-point weight closure for one design."""
    W_TO = W0_guess
    history = [W_TO]
    converged = False

    for _ in range(max_iter):
        evaluation = evaluate_one(W_TO, mission, design)
        if not evaluation["propulsion_valid"]:
            break
        W_calc = evaluation["W_TO_calc"]
        W_new = (1 - relax) * W_TO + relax * W_calc     # damped update
        history.append(W_new)
        if not math.isfinite(W_new) or W_new <= 0 or W_new > 1e4:
            break                                         # diverged: no closure
        if abs(W_new - W_TO) / W_TO < tol:
            W_TO, converged = W_new, True
            break
        W_TO = W_new

    state = evaluate_one(W_TO, mission, design)
    converged = converged and state["propulsion_valid"]
    state["history"] = history
    state["converged"] = converged
    state["feasible"] = converged and W_TO <= mission.W_max
    return state


def _closed_weight(x, mission: MissionCase):
    """Objective for the optimizer: closed W_TO, big penalty if it won't close.
    W_max is not enforced here, so a design that is too heavy is still reported
    (print_report flags it NOT CLOSED)."""
    r = size_drone(mission, Design(*x))
    return r["W_TO"] if r["converged"] else 1e3


def sweep_designs(mission: MissionCase, n_sigma=25, n_tau=5, n_S=5, n_L=7, n_sw=5):
    """Carpet sweep over altitude, t/c, wing size and fuselage length (diameter at
    its minimum), then polish the best point with diameter free too."""
    sig_grid = np.geomspace(sigma_floor(mission), mission.sig_max, n_sigma)
    tau_grid = np.linspace(mission.tau_min, 0.08, n_tau)
    Sf_grid = np.linspace(1.0, 2.0, n_S)
    L_grid = np.linspace(mission.L_min, mission.L_max, n_L)
    sw_grid = np.linspace(0.0, max_sweep_deg(mission), n_sw)

    best_x, best_W = None, math.inf
    for sig in sig_grid:
        for tau in tau_grid:
            for Sf in Sf_grid:
                for L in L_grid:
                    for sw in sw_grid:
                        x = (sig, tau, Sf, L, mission.d_min, sw)
                        W = _closed_weight(x, mission)
                        if W < best_W:
                            best_x, best_W = x, W
    if best_x is None or best_W >= 1e3:
        raise RuntimeError("No design in the sweep converges")

    # Local polish within the same bounds the GP uses
    bounds = [(sigma_floor(mission), mission.sig_max), (mission.tau_min, 0.2), (1.0, 5.0),
              (mission.L_min, mission.L_max), (mission.d_min, 2.0),
              (0.0, max_sweep_deg(mission))]
    res = minimize(_closed_weight, best_x, args=(mission,), method="L-BFGS-B",
                   bounds=bounds, options={"ftol": 1e-14, "gtol": 1e-10})
    x_opt = res.x if res.fun <= best_W else best_x
    return Design(*x_opt), sig_grid


def drag_at_speed(V, result, mission: MissionCase):
    """Cruise drag vs speed for the sized aircraft at its cruise altitude and
    start-of-cruise weight. Same drag model as evaluate_one, with beta taken
    at each speed's Mach (held at M 1.2 values below 1.2, like the GP)."""
    m = mission
    V = np.asarray(V, dtype=float)
    M = V / result["a"]
    q = 0.5 * result["rho"] * V ** 2
    S, tau = result["S"], result["tau"]
    C_L = result["W_cr_start"] / (q * S)
    terms, _ = drag_terms(m, S=S, tau=tau, c_sw=result["cos_sw"], L=result["L_fus"], d=result["d_fus"],
                          rho=result["rho"], V=V, M=M, C_L=C_L)
    return q * sum(terms.values()) * S


PLOT_DIR = Path(__file__).resolve().parent / "plots"


def plot_path(group: str, name: str) -> Path:
    """plots/<group>/<name>, creating the folder if needed."""
    d = PLOT_DIR / group
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def plot_drag_vs_velocity(result, mission: MissionCase, fname=None):
    a = result["a"]
    V = np.linspace(0.3 * a, 2.0 * a, 400)
    D = drag_at_speed(V, result, mission)
    T_avail = result["lapse"] * result["T_SL"]     # turbojet: ~constant with speed
    V_cr = mission.M * a

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V, D, label="Drag required")
    ax.axhline(T_avail, color="tab:red", linestyle="--",
               label=f"Thrust available = {T_avail:.1f} lbf")
    ax.plot(V_cr, drag_at_speed(V_cr, result, mission), "k*", ms=12,
            label=f"Cruise, M {mission.M:g}")
    ax.axvspan(V[0], 1.2 * a, color="gray", alpha=0.12,
               label="M < 1.2: drag coefficients held at M 1.2 values")
    ax.set_xlabel("Velocity (ft/s)")
    ax.set_ylabel("Force (lbf)")
    ax.set_title(f"Drag and thrust vs velocity at h = {result['h']:,.0f} ft, "
                 f"W = {result['W_cr_start']:.1f} lbf")
    ax.set_ylim(0, 1.05 * D.max())

    top = ax.secondary_xaxis("top", functions=(lambda v: v / a, lambda M: M * a))
    top.set_xlabel("Mach")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname or plot_path("drag", "drag_vs_velocity.png"), dpi=150)
    plt.close(fig)


def wing_planform(S: float, AR: float, sweep_deg: float):
    """Trapezoidal wing planform from area, aspect ratio and leading-edge sweep.

    Straight trailing edge (perpendicular to the centerline), swept leading edge.
    That fixes the chords: c_root = S/b + (b/4) tan(sweep), c_tip = S/b - (b/4)
    tan(sweep). The tip chord reaches zero at tan(sweep) = 4/AR, a pure delta,
    which is the sweep limit used everywhere else in this model.
    """
    b = math.sqrt(AR * S)
    tan_le = math.tan(math.radians(sweep_deg))
    c_root = S / b + 0.25 * b * tan_le
    c_tip = max(S / b - 0.25 * b * tan_le, 0.0)
    mac = 2.0 / 3.0 * (c_root + c_tip - c_root * c_tip / (c_root + c_tip))
    tip_le_x = 0.5 * b * tan_le                      # aft offset of the LE at the tip
    sweep_c4 = math.degrees(math.atan((tip_le_x + 0.25 * (c_tip - c_root)) / (0.5 * b)))
    return {"span": b, "c_root": c_root, "c_tip": c_tip, "taper": c_tip / c_root,
            "mac": mac, "tip_le_x": tip_le_x, "sweep_le": sweep_deg,
            "sweep_c4": sweep_c4, "sweep_te": 0.0}


def print_planform(result, mission: MissionCase):
    p = wing_planform(result["S"], mission.AR, result["sweep_deg"])
    print("Wing planform (trapezoid, straight trailing edge)")
    print("-------------------------------------------------")
    print(f"   Area S           : {result['S']:8.3f} ft^2")
    print(f"   Aspect ratio     : {mission.AR:8.3f}")
    print(f"   Span             : {p['span']:8.3f} ft  ({p['span'] * 12:.1f} in)")
    print(f"   Root chord       : {p['c_root']:8.3f} ft  ({p['c_root'] * 12:.1f} in)")
    print(f"   Tip chord        : {p['c_tip']:8.3f} ft  ({p['c_tip'] * 12:.1f} in)")
    print(f"   Taper ratio      : {p['taper']:8.3f}" + ("   (delta)" if p["c_tip"] < 1e-9 else ""))
    print(f"   Mean aero chord  : {p['mac']:8.3f} ft  ({p['mac'] * 12:.1f} in)")
    print(f"   Sweep, LE / c/4  : {p['sweep_le']:8.2f} / {p['sweep_c4']:.2f} deg")
    print(f"   Thickness t/c    : {result['tau']:8.4f}  (max thickness {result['tau'] * p['mac'] * 12:.2f} in at the MAC)")
    print()


def plot_planform(result, mission: MissionCase, fname=None):
    """Top view of the wing, nose up, with the key dimensions."""
    p = wing_planform(result["S"], mission.AR, result["sweep_deg"])
    b, cr, ct, xt = p["span"], p["c_root"], p["c_tip"], p["tip_le_x"]
    # (spanwise y, aft x): right wing, then mirror
    right = [(0, 0), (b / 2, xt), (b / 2, xt + ct), (0, cr)]
    left = [(-y, x) for y, x in reversed(right)]
    pts = left + right

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.fill([y for y, _ in pts], [x for _, x in pts], color="#2a73d4", alpha=0.18)
    ax.plot([y for y, _ in pts] + [pts[0][0]], [x for _, x in pts] + [pts[0][1]],
            color="#2a73d4", lw=2)
    ink = "#59636e"
    ax.axvline(0, color=ink, lw=0.8, ls=":")
    # dimensions
    ax.annotate("", xy=(-b / 2, cr + 0.12 * b), xytext=(b / 2, cr + 0.12 * b),
                arrowprops=dict(arrowstyle="<->", color=ink))
    ax.text(0, cr + 0.15 * b, f"span {b:.2f} ft ({b * 12:.1f} in)", ha="center", va="top", color=ink)
    ax.annotate("", xy=(0.04 * b, 0), xytext=(0.04 * b, cr), arrowprops=dict(arrowstyle="<->", color=ink))
    ax.text(0.06 * b, cr / 2, f"root chord\n{cr * 12:.1f} in", va="center", color=ink, fontsize=9)
    if ct > 1e-9:
        ax.text(b / 2 + 0.03 * b, xt + ct / 2, f"tip chord\n{ct * 12:.1f} in", va="center", color=ink, fontsize=9)
    ax.text(-b / 2, xt * 0.5, f"LE sweep {p['sweep_le']:.1f}°", ha="right", color=ink, fontsize=9)
    ax.set_aspect("equal")
    ax.set_ylim(cr + 0.25 * b, -0.08 * b)   # reversed: nose up, room below for the span dimension
    ax.set_xlim(-0.75 * b, 0.75 * b)
    ax.set_xlabel("Spanwise (ft)", color=ink)
    ax.set_ylabel("Aft of root leading edge (ft)", color=ink)
    ax.set_title(f"Wing planform: S = {result['S']:.2f} ft², AR {mission.AR:g}, taper {p['taper']:.2f}, "
                 f"t/c {result['tau']:.3f}", loc="left", color="#1f2328", fontsize=10)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#e6e6e6")
    ax.tick_params(colors=ink, length=0)
    fig.tight_layout()
    fig.savefig(fname or plot_path("wing", "wing_planform.png"), dpi=150)
    plt.close(fig)


def print_report(result, design: Design, mission: MissionCase):
    """Same layout as the lcsolver summary in mission_sim.py."""
    print("Report")
    print("------")
    print("   Traditional weight closure (damped fixed-point iteration)")
    print("   Design sweep over altitude, t/c and wing area + L-BFGS-B polish")
    print(f"   Engine: {mission.engine_model or 'continuous fit (conceptual estimate; engine may shrink)'}")
    print(f"   Thrust lapse: {'user scenario, T/T_SLS = ' + str(mission.lapse_override) if mission.lapse_override is not None else 'screening fit; not a validated off-design map'}")
    print()
    print("Objective")
    print("---------")
    print(f"   {result['W_TO']:.4f} lbf")
    print()
    print("Variables")
    print("---------")
    units = {"a": "ft/s", "c_T": "1/hr", "D": "lbf", "h": "ft", "rho": "slug/ft**3",
             "S": "ft**2", "T_SL": "lbf", "T_seg": "lbf", "P_req": "hp"}
    for key in ["a", "C_D", "C_D0", "C_Dw", "C_L", "D", "h", "lapse", "rho",
                "S", "T_SL", "tau", "c_T", "W_cr_end", "W_cr_start", "W_empty",
                "W_eng", "W_f_climb", "W_f_cruise", "W_f_res", "W_other", "W_fus",
                "W_TO", "W_wing", "W_ZF", "T_seg", "P_req"]:
        unit = units.get(key, "lbf" if key.startswith("W_") else "-")
        print(f"   {key:10s} : {result[key]:12.4f}   [{unit}]")
    print()
    print("Design choices")
    print("--------------")
    print(f"   sigma      : {design.sigma:12.4f}   (bounds {sigma_floor(mission):.4f} - {mission.sig_max}, h <= {mission.h_max:,.0f} ft)")
    print(f"   tau        : {design.tau:12.4f}   (min {mission.tau_min})")
    print(f"   S / S_stall: {design.S_factor:12.4f}")
    print(f"   sweep [deg]: {design.sweep_deg:12.4f}   (max {max_sweep_deg(mission):.1f} = delta with AR {mission.AR:g})")
    print(f"   L_fus      : {design.L_fus:12.4f}   [ft]  (bounds {mission.L_min:g} - {mission.L_max:g})")
    print(f"   d_fus      : {result['d_fus']:12.4f}   [ft]  (engine envelope: {'catalogue' if mission.engine_model else 'weight fit'}, {result['d_fus'] * 12:.1f} in)")
    print(f"   Swet_fus   : {result['Swet_fus']:12.4f}   [ft^2]")
    print(f"   CDA_fus_w  : {result['CDA_fus_w']:12.4f}   [ft^2]")
    sizing = ("acceleration path at Mach %.2f" % result["M_path"]
              if result["T_SL_path"] >= max(result["T_SL"], result["T_seg"] / result["lapse"]) * (1 - 1e-9)
              and result["T_SL_path"] > mission.TW_TO * result["W_TO"]
              else "Mach 1 segment thrust" if result["T_SL"] > mission.TW_TO * result["W_TO"] + 1e-9 else "takeoff T/W")
    print(f"   Engine sized by {sizing}")
    print()
    print_planform(result, mission)
    from breakdown import print_breakdown
    print_breakdown(result, {k: getattr(mission, k) for k in (
        "W_pay", "W_prop_install", "W_fixed", "f_other", "f_TO", "f_res", "M", "Swet_S", "E_WD", "K_ind", "TW_TO", "T_int", "T_slope",
        "t_super", "w_fus", "wing_weight_factor", "fuselage_weight_factor", "w_surf", "cf_A", "cf_n", "mu", "k_fin", "AR", "AR_fin", "tau_fin", "sweep_fin_deg",
        "k_fric", "Q_fin", "f_strake", "AR_str", "sweep_str_deg", "C_pb", "base_open", "CD_pod", "k_pod", "f_int")})
    print("Constants")
    print("---------")
    for fld in fields(mission):
        value = getattr(mission, fld.name)
        print(f"   {fld.name:10s} : {value if value is None or isinstance(value, str) else format(value, '.6g')}")
    print()
    print(f"Fuel fraction   : {(result['W_f_climb'] + result['W_f_cruise'] + result['W_f_res']) / result['W_TO']:.3f}")
    print(f"Empty fraction  : {result['W_empty'] / result['W_TO']:.3f}")
    print(f"Segment L/D      : {result['C_L'] / result['C_D']:.2f}")
    print(f"Closure residual: {abs(result['W_TO_calc'] - result['W_TO']):.2e} lbf")
    print("\nPost Solve Report")
    print("-----------------")
    status = "converged and within weight limit" if result["feasible"] else (
        "converged; exceeds weight limit" if result["converged"] else "NOT CLOSED")
    print(f"   Status: {status}")
    if result.get("propulsion_error"):
        print(f"   Propulsion: {result['propulsion_error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mission design sweep with optional fixed catalogue engine")
    add_hardware_arguments(parser)
    parser.add_argument("--continuous-engine-fit", action="store_true", help="explicit conceptual engine resizing")
    parser.add_argument("--max-weight", type=float, default=80)
    parser.add_argument("--accel-margin", type=float, default=0.10)
    parser.add_argument("--wing-weight-factor",type=float,default=MissionCase.wing_weight_factor)
    parser.add_argument("--fuselage-weight-factor",type=float,default=MissionCase.fuselage_weight_factor)
    parser.set_defaults(engine=None)
    args = parser.parse_args()
    if args.continuous_engine_fit:
        args.engine = None
    if args.engine:
        try:
            catalogue_engine(args.engine)
        except ValueError as error:
            parser.error(str(error))
    mission = MissionCase(engine_model=args.engine, W_prop_install=args.installation_weight,
                          W_pay=args.payload_weight, lapse_override=args.lapse,
                          W_max=args.max_weight, accel_margin=args.accel_margin, L_min=6.0,
                          wing_weight_factor=args.wing_weight_factor,fuselage_weight_factor=args.fuselage_weight_factor)
    design, sig_grid = sweep_designs(mission)
    result = size_drone(mission, design)
    print_report(result, design, mission)

    # Convergence history of the final closure
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(result["history"])), result["history"], marker="o", ms=3)
    plt.axhline(mission.W_max, color="red", linestyle="--", label=f"W_max = {mission.W_max:g} lb")
    plt.xlabel("Iteration")
    plt.ylabel("W_TO (lbf)")
    plt.title("Weight closure convergence (optimal design)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("mission", "weight_convergence.png"), dpi=150)
    plt.close()

    # Altitude sweep at the optimal t/c and wing size
    h_list, W_list = [], []
    for sig in sig_grid:
        r = size_drone(mission, Design(sig, design.tau, design.S_factor, design.L_fus, design.d_fus, design.sweep_deg))
        if r["converged"]:
            h_list.append(r["h"])
            W_list.append(r["W_TO"])
    plt.figure(figsize=(8, 5))
    plt.plot(h_list, W_list, marker="o", ms=3, label="Closed W_TO")
    plt.plot(result["h"], result["W_TO"], "k*", ms=12, label="Optimum")
    plt.axhline(mission.W_max, color="red", linestyle="--", label=f"W_max = {mission.W_max:g} lb")
    plt.xlabel("Cruise altitude (ft)")
    plt.ylabel("W_TO (lbf)")
    plt.title("Mission sweep: closed takeoff weight vs cruise altitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("mission", "mission_sweep.png"), dpi=150)
    plt.close()

    plot_drag_vs_velocity(result, mission)
    plot_planform(result, mission)

    print("\nPlots saved to plots/mission/ (weight_convergence, mission_sweep), plots/drag/drag_vs_velocity.png and plots/wing/wing_planform.png")

