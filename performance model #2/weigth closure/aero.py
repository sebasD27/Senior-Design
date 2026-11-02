# ===========
# Description
# ===========
# Component drag build-up shared by missionsweep.py and breakdown.py. It is the
# same model written symbolically in mission_sim.py; breakdown.py prints the GP
# solution through this function and checks the two agree.
#
#   skin friction per component: C_f = cf_A Re^-cf_n x Raymer compressibility factor,
#       Re = rho V l / mu  with l the component length (mean chord for lifting
#       surfaces, length for the fuselage), times a form factor FF and an
#       interference factor Q (Raymer, Aircraft Design: A Conceptual Approach):
#         wing / strake : FF = [1 + 0.6/(x/c)m t/c + 100 (t/c)^4] 1.34 M^0.18 cos(sweep)^0.28
#         fin           : same expression, fixed t/c and sweep
#         fuselage      : FF = 1 + 60/f^3 + f/400,  f = L/d
#   wave drag: wing and fin thickness (K_wave tau^2 cos^2 sweep), fuselage Sears-Haack area
#   base drag: C_pb x open base area (fuselage base minus the exhaust nozzle), pi d^2/4 base_open
#   inlet / ventral pod: C_D,pod x frontal area k_pod (pi d^2/4), on top of the Sears-Haack fuselage
#   interference + excrescences: f_int x (friction + wave + base + pod)
#   lift-dependent drag: K_ind K_sup C_L^2
#   vertical fin sized from the fuselage side area, S_v = k_fin L d (a slender body is destabilizing,
#   so wing-based tail volume gives a uselessly small fin here); elevons: no horizontal tail

import numpy as np

XC_M = 0.3                      # chordwise position of max thickness (Raymer's (x/c)m)

# Transonic wave-drag rise. The Ackeret/Sears-Haack wave terms below are the M >= 1.2 (supersonic) level.
# Below that they are scaled by F_tr(M): zero up to M 0.8, rising to TR_PEAK times the supersonic level at
# M 1.05 (the drag-rise peak), then back to 1 at M 1.2. TR_PEAK = 1.25 makes F_tr(1.0) = 1 exactly, so the
# Mach 1 dash drag (and the friction calibration against the hand estimate) is unchanged. The shape is a
# PLACEHOLDER chosen by hand, not a CFD or wind-tunnel result; swap in measured values when available.
TR_M = (0.8, 1.05, 1.2)
TR_PEAK = 1.25


def transonic_factor(M, peak=TR_PEAK):
    """Multiplier on the supersonic-level wave drag at Mach M (1 at M >= 1.2)."""
    return np.interp(np.asarray(M, dtype=float), TR_M, (0.0, peak, 1.0))


def comp_factor(M):
    """Raymer's flat-plate compressibility correction on C_f (1 at M 0)."""
    return (1.0 + 0.144 * np.asarray(M, dtype=float) ** 2) ** -0.65


def surface_ff(tau, M, cos_sweep):
    """Raymer wing/tail form factor."""
    return (1.0 + 0.6 / XC_M * tau + 100.0 * tau ** 4) * 1.34 * np.asarray(M, dtype=float) ** 0.18 * cos_sweep ** 0.28


def fin_area(c, L, d):
    """Vertical fin planform area as a fraction of the fuselage side area L d."""
    return c.k_fin * L * d


def drag_terms(c, *, S, tau, c_sw, L, d, rho, V, M, C_L):
    """C_D contributions (relative to wing area S). c: constants (MissionCase or namespace)."""
    V = np.asarray(V, dtype=float)
    M = np.asarray(M, dtype=float)
    beta = np.sqrt(np.maximum(M, 1.2) ** 2 - 1.0)       # held at M 1.2 values below 1.2
    K_sup, K_wave = beta / 4.0, 4.0 / beta
    kc = comp_factor(M)

    def cf(length):
        return c.cf_A * (rho * V * length / c.mu) ** -c.cf_n * kc

    S_v = fin_area(c, L, d)
    S_str = c.f_strake * S
    cos_fin = np.cos(np.radians(c.sweep_fin_deg))
    cos_str = np.cos(np.radians(c.sweep_str_deg))
    f = L / d

    t = {}
    k = c.k_fric                # calibration multiplier on all skin friction, see MissionCase
    # Fractional powers also accept normalized symbolic geometry in mission_sim.
    # The numerical equations and component coefficients are unchanged.
    t["Skin friction: wing"] = k * cf((S / c.AR)**0.5) * surface_ff(tau, M, c_sw) * c.Swet_S
    t["Skin friction: strake"] = k * (cf((S_str / c.AR_str)**0.5) * surface_ff(tau, M, cos_str)
                                  * c.Swet_S * S_str / S)
    t["Skin friction: fuselage"] = k * cf(L) * (1.0 + 60.0 / f ** 3 + f / 400.0) * np.pi * d * L / S
    t["Skin friction: fin"] = k * (cf((S_v / c.AR_fin)**0.5) * surface_ff(c.tau_fin, M, cos_fin)
                               * c.Q_fin * c.Swet_S * S_v / S)
    CDA_fus = c.E_WD * 4.5 * np.pi * (np.pi * d ** 2 / 4.0 / L) ** 2
    f_tr = transonic_factor(M)
    t["Wave: wing thickness"] = f_tr * K_wave * tau ** 2 * c_sw ** 2
    t["Wave: fin thickness"] = f_tr * K_wave * c.tau_fin ** 2 * cos_fin ** 2 * S_v / S
    t["Wave: fuselage"] = f_tr * CDA_fus / S
    A_fus = np.pi * d ** 2 / 4.0
    t["Base drag"] = c.C_pb * c.base_open * A_fus / S
    t["Inlet / pod"] = c.CD_pod * c.k_pod * A_fus / S
    t["Interference + excrescences"] = c.f_int * sum(t.values())
    lift_coefficient = np.asarray(C_L, dtype=float) if isinstance(C_L, (list, tuple, np.ndarray)) else C_L
    t["Lift-dependent"] = c.K_ind * K_sup * lift_coefficient ** 2
    return t, {"S_v": S_v, "S_str": S_str, "CDA_fus": CDA_fus}
