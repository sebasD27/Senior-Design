# ===========
# Description
# ===========
# Input file for the lcsolver GP optimizer (mission_sim.py). Edit the values below, then run
#     python inputs.py
# from this folder with the esp312 environment. Every knob is a MissionCase field
# (missionsweep.py); a misspelled name stops the run instead of being ignored. Anything not
# listed here keeps its MissionCase default.
#
# Units: lb (weight), lbf (force), ft, ft/s, ft^2, ft^3, s, deg. Altitudes are geometric ft.

from mission_sim import run_case
from missionsweep import MissionCase

# ======================
# Mission and run setup
# ======================
ALTITUDE = 17000.0       # lowest dash altitude under 55 lb within q_max 800 psf (P650, studies/weight_vs_altitude.py)
#   dash altitude, ft. None: the GP chooses it (troposphere only, within the
                         # limits below), then re-solves the exact model there. A number pins it.
ENGINE = "P650-PRO-S-R"  # pinned: closes under 55 lb down to 15,000 ft (P400: 30,000 ft)
#   None: continuous engine fits. "auto": size around every catalogue engine and
                         # keep the lightest aircraft. A name, e.g. "P1000-PRO": pin that engine.

MISSION = dict(
    W_max=55.0,          # lb, takeoff weight cap (Part 107)
    W_pay=0.0,           # lb, payload
    M=1.0,               # was 1.05 (sits on the CFD drag-rise peak)
#   dash Mach
    t_super=5.0,         # s, level dash time at M
    h_min=0.0,           # ft, lowest allowed dash altitude
    h_max=50000.0,       # ft, highest allowed dash altitude (free-altitude GP stops at 36,152 ft)
    V_climb=600.0,       # was 300
#   ft/s, climb speed
    accel_margin=0.05,   # minimum (installed thrust - drag) / drag along climb and acceleration
    f_TO=0.02,           # start, taxi and takeoff fuel / W_TO
    f_idle=0.20,         # idle / full-thrust fuel flow at sea level (0 = engine-off glide). PLACEHOLDER
    RoD_desc=100.0,      # ft/s, idle descent rate
    f_land=0.005,        # approach and landing fuel / zero-fuel weight
    f_res=0.06,          # reserve fuel / zero-fuel weight
    q_max=800.0,         # does not bind at 30,000 ft (q ~440 psf)
#   psf, highest dash dynamic pressure the structure is designed for (Mach 1 at
                         # 20,000 ft is about 680 psf, at sea level about 1,480). PLACEHOLDER
)

# ===========================
# Low speed: wing sizing, takeoff
# ===========================
LOW_SPEED = dict(
    V_stall=120.0,       # ft/s, stall speed at takeoff weight; sizes the wing. No design closes below
                         # about 110 ft/s with C_Lmax 1.0 (see README)
    C_Lmax=1.3,          # usable maximum lift coefficient with delta vortex lift (needs a high landing
                         # attitude: check gear length / tail-strike angle)
    TW_TO=0.5,           # minimum takeoff thrust / weight (ignored with rail_launch)
    rail_launch=True,    # rail launch, landing on legs: wing sized by stall at landing weight
)

# ==================
# Geometry limits
# ==================
GEOMETRY = dict(
    AR=2.0,              # wing aspect ratio
    tau_min=0.03,        # was 0.05
#   minimum wing t/c (the SU2 Euler runs used 0.05)
    L_min=6.0,           # ft, fuselage length limits
    L_max=12.0,
    d_min=0.25,          # ft, minimum fuselage diameter (the engine diameter also sets a floor)
    d_clear=0.04,        # ft, radial clearance engine -> outer skin (casing, mounts, shell)
)

# =============
# Propulsion
# =============
PROPULSION = dict(
    # Catalogue fits (propulsion/engine_catalogue.csv), engine weight W in lb
    T_slope=7.1, T_int=28.6,              # sea-level static thrust = T_slope W + T_int, lbf
    tsfc_int=1.617, tsfc_slope=-0.0117,   # TSFC = tsfc_int + tsfc_slope W, 1/hr
    W_eng_min=4.0,       # lb, lightest catalogue engine; the fits are invalid below
    W_eng_max=43.21,     # lb, heaviest catalogue engine
    d_coef=2.5282 / 12, d_exp=0.3969,     # engine diameter = d_coef W^d_exp, ft
    # Thrust lapse at Mach 1 from the engine sim: lapse = K_ram sigma^lapse_exp. Conservative fit: inlet
    # recovery 0.98 static, 0.90 at Mach 1 (the lossless-inlet fit was K_ram 1.266656, exponent 0.946451)
    K_ram=1.10714,
    lapse_exp=0.93958,
    eta_inst=0.97,       # installed / uninstalled thrust: nozzle, power extraction, duct distortion. PLACEHOLDER
    lapse_override=None, # a single thrust / SLS-thrust number used along the whole path, or None
    W_prop_install=0.0,  # lb, installation hardware not already in the engine or other weights
)

# =============
# Aerodynamics
# =============
AERO = dict(
    k_fric=0.613,        # was 1 (hand-estimate calibration; confirm with viscous CFD)
#   skin-friction calibration to the hand estimate (1.0 = plain Raymer build-up)
    K_ind=2.0,           # lift-dependent drag factor (fitted to the Euler sweep: K = 0.33)
    E_WD=1.2,            # fuselage wave drag / Sears-Haack (close-to-Sears-Haack, area-ruled body)
    Swet_S=2.05,         # wing wetted area / planform area
    # Fin and strakes. PLACEHOLDERS: set from the CAD
    k_fin=0.25,          # fin area / (fuselage length x diameter)
    AR_fin=1.2, tau_fin=0.04, sweep_fin_deg=50.0, Q_fin=1.05,
    f_strake=0.15,       # strake area / wing area
    AR_str=0.6, sweep_str_deg=70.0,
    # Base, inlet pod and interference. PLACEHOLDERS: set from the CAD
    C_pb=0.15,           # base pressure coefficient on the open base area
    base_open=0.05,      # open base / fuselage frontal: boattail closes onto the nozzle, exhaust fills it
    CD_pod=0.2,          # pod drag coefficient on its frontal area
    k_pod=0.3,           # pod frontal area / fuselage frontal area
    f_int=0.10,          # interference and excrescences, fraction of the other parasite drag
)

# ===========================
# Structures and fixed weights
# ===========================
STRUCTURES = dict(
    N_ult=6.0,           # ultimate load factor
    K_wing=0.0103,       # Raymer fighter wing coefficient
    wing_weight_factor=2.0,      # allowance on the Raymer wing estimate
    fuselage_weight_factor=1.5,  # allowance on the shell and frame estimate
    w_fus=0.35,          # lb/ft^2 of fuselage wetted area (CFRP shell and frames)
    w_surf=0.40,         # lb/ft^2 of fin and strake planform
    W_fixed=3.0,         # lb, avionics, datalink, battery, servos
    f_other=0.20,        # other empty weight / W_TO
    W_wing_min=0.0,      # lb, optional floors from measured or analyzed parts
    W_fus_min=0.0,
)

# ================
# Internal volume
# ================
VOLUME = dict(
    f_vol_use=0.6,       # usable fraction of the Sears-Haack fuselage volume. PLACEHOLDER
    eng_LD=2.5,          # engine length / diameter (the catalogue has no lengths). PLACEHOLDER
    rho_fuel=50.0,       # lb/ft^3, Jet-A
    k_tank=1.1,          # tank volume / fuel volume
    V_sys=0.05,          # ft^3, avionics, battery, servos, pump, ECU. PLACEHOLDER
    f_duct=0.05,         # inlet duct volume / (engine frontal area x fuselage length); from testDrone.csm
)


def mission_case():
    """All the dictionaries above as one MissionCase."""
    inputs = {**MISSION, **LOW_SPEED, **GEOMETRY, **PROPULSION, **AERO, **STRUCTURES, **VOLUME}
    return MissionCase(engine_model=ENGINE, **inputs)


if __name__ == "__main__":
    raise SystemExit(run_case(mission_case(), ALTITUDE))
