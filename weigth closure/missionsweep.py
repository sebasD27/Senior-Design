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
from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class MissionCase:
    # Mission requirements
    W_max: float = 55.0          # lbf, max takeoff weight (Part 107)
    W_pay: float = 0.0           # lbf, payload
    t_super: float = 5.0         # s, level supersonic flight time
    M: float = 1.0               # cruise Mach (drag held at M 1.2 values below 1.2)
    rho_SL: float = 2.377e-3     # slug/ft^3
    V_stall: float = 120.0       # ft/s
    C_Lmax: float = 1.0
    TW_TO: float = 0.5           # required takeoff T/W
    # Propulsion
    c_T: float = 1.25             # 1/hr, cruise TSFC
    K_ram: float = 0.5           # ram recovery factor on thrust lapse
    TW_eng: float = 9.0          # installed engine T/W
    # Aerodynamics
    C_f: float = 0.003
    Swet_S: float = 2.05
    Swet_fus: float = 12.0       # ft^2
    CDA_fus_w: float = 0.086     # ft^2
    K_ind: float = 1.2
    # Structures / weights
    AR: float = 2.0
    N_ult: float = 6.0
    K_wing: float = 0.0103 * 2.0
    tau_min: float = 0.025
    f_other: float = 0.20
    W_fixed: float = 3.0         # lbf
    f_TO: float = 0.02
    V_climb: float = 600.0       # ft/s
    eta_climb: float = 0.5
    g: float = 32.174            # ft/s^2
    # Atmosphere fit (same as the GP)
    h_fit: float = 17335.8       # ft
    a_SL: float = 1116.4         # ft/s
    a_strat: float = 968.1       # ft/s
    sig_max: float = 0.4481
    sig_min: float = 0.1197
    h_max: float = 40000.0       # ft, max cruise altitude
    f_res: float = 0.06


def sigma_floor(m: MissionCase) -> float:
    """Lowest allowed density ratio: the tighter of sig_min and the altitude
    ceiling, h = h_fit * sigma^-0.5701 <= h_max (same fit as the GP)."""
    return max(m.sig_min, (m.h_max / m.h_fit) ** (-1 / 0.5701))


@dataclass
class Design:
    """The free design choices the GP gets to optimize."""
    sigma: float                 # cruise density ratio (sets altitude)
    tau: float                   # wing t/c
    S_factor: float = 1.0        # wing area / stall-limited minimum area


def evaluate_one(W_TO: float, mission: MissionCase, design: Design):
    """Fly the mission once at a guessed W_TO. Same names as mission_sim.py."""
    m = mission
    sigma, tau = design.sigma, design.tau

    # Atmosphere
    rho = m.rho_SL * sigma
    h = m.h_fit * sigma ** -0.5701
    a = max(m.a_strat, m.a_SL * sigma ** 0.1175)
    V = m.M * a
    q = 0.5 * rho * V ** 2
    lapse = m.K_ram * sigma ** 0.7

    beta = math.sqrt(max(m.M, 1.2) ** 2 - 1.0)
    K_sup = beta / 4.0
    K_wave = 4.0 / beta

    # Wing sized by stall speed (S_factor >= 1 allows a bigger wing)
    S = design.S_factor * 2.0 * W_TO / (m.rho_SL * m.V_stall ** 2 * m.C_Lmax)

    # Climb / accel fuel (energy method), c_T converted 1/hr -> 1/s
    W_f_climb = (m.f_TO * W_TO
                 + (m.c_T / 3600.0) * W_TO * (h + V ** 2 / (2.0 * m.g))
                 / (m.V_climb * m.eta_climb))
    W_cr_start = W_TO - W_f_climb

    # Cruise aerodynamics (lift = weight at start of the supersonic leg)
    C_L = W_cr_start / (q * S)
    C_D0 = m.C_f * m.Swet_S + m.C_f * m.Swet_fus / S
    C_Dw = K_wave * tau ** 2 + m.CDA_fus_w / S
    C_D = C_D0 + C_Dw + m.K_ind * K_sup * C_L ** 2
    D = q * C_D * S

    # Engine sized by the worse of takeoff T/W and cruise thrust
    T_SL = max(m.TW_TO * W_TO, D / lapse)

    # Empty weight build-up
    W_wing = (m.K_wing * math.sqrt(m.N_ult * W_TO) * S ** 0.622
              * m.AR ** 0.785 * tau ** -0.4)
    W_eng = T_SL / m.TW_eng
    W_other = m.W_fixed + m.f_other * W_TO
    W_empty = W_wing + W_eng + W_other
    W_ZF = m.W_pay + W_empty

    # Reserve and Breguet supersonic leg
    W_f_res = m.f_res * W_ZF
    W_cr_end = W_ZF + W_f_res
    z_bre = (m.t_super / 3600.0) * m.c_T * C_D / C_L
    W_f_cruise = W_cr_end * (z_bre + z_bre ** 2 / 2 + z_bre ** 3 / 6 + z_bre ** 4 / 24)

    return {
        "W_TO": W_TO, "W_ZF": W_ZF, "W_empty": W_empty, "W_wing": W_wing,
        "W_eng": W_eng, "W_other": W_other, "W_f_climb": W_f_climb,
        "W_f_cruise": W_f_cruise, "W_f_res": W_f_res, "W_cr_start": W_cr_start,
        "W_cr_end": W_cr_end, "S": S, "tau": tau, "C_L": C_L, "C_D": C_D,
        "C_D0": C_D0, "C_Dw": C_Dw, "D": D, "T_SL": T_SL, "z_bre": z_bre,
        "h": h, "rho": rho, "a": a, "lapse": lapse,
        # build-up: what the mission says W_TO has to be
        "W_TO_calc": W_ZF + W_f_climb + W_f_cruise + W_f_res,
    }


def size_drone(mission: MissionCase, design: Design, W0_guess: float = 50.0,
               relax: float = 0.5, tol: float = 1e-10, max_iter: int = 500):
    """Traditional fixed-point weight closure for one design."""
    W_TO = W0_guess
    history = [W_TO]
    converged = False

    for _ in range(max_iter):
        W_calc = evaluate_one(W_TO, mission, design)["W_TO_calc"]
        W_new = (1 - relax) * W_TO + relax * W_calc     # damped update
        history.append(W_new)
        if not math.isfinite(W_new) or W_new <= 0 or W_new > 1e4:
            break                                         # diverged: no closure
        if abs(W_new - W_TO) / W_TO < tol:
            W_TO, converged = W_new, True
            break
        W_TO = W_new

    state = evaluate_one(W_TO, mission, design)
    state["history"] = history
    state["converged"] = converged
    state["feasible"] = converged and W_TO <= mission.W_max
    return state


def _closed_weight(x, mission: MissionCase):
    """Objective for the optimizer: closed W_TO, big penalty if it won't close."""
    r = size_drone(mission, Design(*x))
    return r["W_TO"] if r["feasible"] else 1e3


def sweep_designs(mission: MissionCase, n_sigma=25, n_tau=9, n_S=9): #NO idea how this works
    """Carpet sweep over altitude, t/c and wing size, then polish the best point."""
    sig_grid = np.geomspace(sigma_floor(mission), mission.sig_max, n_sigma)
    tau_grid = np.linspace(mission.tau_min, 0.08, n_tau)
    Sf_grid = np.linspace(1.0, 2.0, n_S)

    best_x, best_W = None, math.inf
    for sig in sig_grid:
        for tau in tau_grid:
            for Sf in Sf_grid:
                W = _closed_weight((sig, tau, Sf), mission)
                if W < best_W:
                    best_x, best_W = (sig, tau, Sf), W
    if best_x is None or best_W >= mission.W_max:
        raise RuntimeError("No design in the sweep closes under W_max")

    # Local polish within the same bounds the GP uses
    bounds = [(sigma_floor(mission), mission.sig_max), (mission.tau_min, 0.2), (1.0, 5.0)]
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
    beta = np.sqrt(np.maximum(M, 1.2) ** 2 - 1.0)
    K_sup, K_wave = beta / 4.0, 4.0 / beta
    q = 0.5 * result["rho"] * V ** 2
    S, tau = result["S"], result["tau"]
    C_L = result["W_cr_start"] / (q * S)
    C_D = (m.C_f * m.Swet_S + m.C_f * m.Swet_fus / S
           + K_wave * tau ** 2 + m.CDA_fus_w / S
           + m.K_ind * K_sup * C_L ** 2)
    return q * C_D * S


def plot_drag_vs_velocity(result, mission: MissionCase, fname="drag_vs_velocity.png"):
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
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def print_report(result, design: Design, mission: MissionCase):
    """Same layout as the lcsolver summary in mission_sim.py."""
    print("Report")
    print("------")
    print("   Traditional weight closure (damped fixed-point iteration)")
    print("   Design sweep over altitude, t/c and wing area + L-BFGS-B polish")
    print()
    print("Objective")
    print("---------")
    print(f"   {result['W_TO']:.4f} lbf")
    print()
    print("Variables")
    print("---------")
    units = {"a": "ft/s", "D": "lbf", "h": "ft", "rho": "slug/ft**3",
             "S": "ft**2", "T_SL": "lbf"}
    for key in ["a", "C_D", "C_D0", "C_Dw", "C_L", "D", "h", "lapse", "rho",
                "S", "T_SL", "tau", "W_cr_end", "W_cr_start", "W_empty",
                "W_eng", "W_f_climb", "W_f_cruise", "W_f_res", "W_other",
                "W_TO", "W_wing", "W_ZF", "z_bre"]:
        unit = units.get(key, "lbf" if key.startswith("W_") else "-")
        print(f"   {key:10s} : {result[key]:12.4f}   [{unit}]")
    print()
    print("Design choices")
    print("--------------")
    print(f"   sigma      : {design.sigma:12.4f}   (bounds {sigma_floor(mission):.4f} - {mission.sig_max}, h <= {mission.h_max:,.0f} ft)")
    print(f"   tau        : {design.tau:12.4f}   (min {mission.tau_min})")
    print(f"   S / S_stall: {design.S_factor:12.4f}")
    sizing = "cruise thrust" if result["T_SL"] > mission.TW_TO * result["W_TO"] + 1e-9 else "takeoff T/W"
    print(f"   Engine sized by {sizing}")
    print()
    print("Constants")
    print("---------")
    for fld in fields(mission):
        print(f"   {fld.name:10s} : {getattr(mission, fld.name):12.6g}")
    print()
    print(f"Fuel fraction   : {(result['W_f_climb'] + result['W_f_cruise'] + result['W_f_res']) / result['W_TO']:.3f}")
    print(f"Empty fraction  : {result['W_empty'] / result['W_TO']:.3f}")
    print(f"Cruise L/D      : {result['C_L'] / result['C_D']:.2f}")
    print(f"Closure residual: {abs(result['W_TO_calc'] - result['W_TO']):.2e} lbf")
    print("\nPost Solve Report")
    print("-----------------")
    print(f"   Status: {'converged' if result['feasible'] else 'NOT CLOSED'}")


if __name__ == "__main__":
    mission = MissionCase()
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
    plt.savefig("weight_convergence.png", dpi=150)
    plt.close()

    # Altitude sweep at the optimal t/c and wing size
    h_list, W_list = [], []
    for sig in sig_grid:
        r = size_drone(mission, Design(sig, design.tau, design.S_factor))
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
    plt.savefig("mission_sweep.png", dpi=150)
    plt.close()

    plot_drag_vs_velocity(result, mission)

    print("\nPlots saved to weight_convergence.png, mission_sweep.png and drag_vs_velocity.png")
