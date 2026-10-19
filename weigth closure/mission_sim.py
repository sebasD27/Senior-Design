# ===========
# Description
# ===========
# Weight closure for a fuel-powered supersonic drone (<= 55 lb), formulated as a
# Geometric Program and solved with lcsolver.
#
# Objective: minimize takeoff gross weight W_TO.
#
# Mission:  takeoff / climb / accelerate  ->  5 s of level supersonic flight
#           (power balance, optional climb angle)  ->  descent / landing with
#           reserve fuel
#
# Models:
#   - Weight build-up: payload + empty weight + fuel, capped at W_max
#   - Empty weight: wing (Raymer fighter wing fit) + engines (engine T/W)
#     + everything else as a fraction of W_TO plus a fixed weight
#   - Supersonic drag: skin friction + wing wave drag (thin airfoil)
#     + fuselage wave drag (Sears-Haack area) + lift-dependent drag
#   - Supersonic segment: level flight, sized for power. Thrust T = D, fuel =
#     c_T T t (linear, so no Breguet/Taylor series), engine sized by T / lapse.
#     (Climbing through the segment and diving were both studied: level flight
#     gives the lightest aircraft, and diving on gravity is not allowed.)
#   - Wing sizing: landing stall speed
#   - Engine sizing: takeoff T/W and cruise thrust with a thrust lapse
#     that falls with air density (turbojet, T ~ sigma^0.7)
#   - Altitude is a design variable: thinner air cuts drag but also cuts
#     thrust and costs more climb fuel (energy method)
#
# Every GP rule: each constraint must be  posynomial <= monomial
# (or monomial == monomial). Keep new constraints in that form and
# lcsolver will keep detecting a GP.

# =================
# Import Statements
# =================
from typing import cast

import sys

import numpy as np

import lcsolver
from lcsolver import Formulation

# ===================
# Declare Formulation
# ===================
# cast: Pyomo's Block.__new__ confuses Pylance into typing f as IndexedBlock
f = cast(Formulation, Formulation())

# =================
# Declare Variables
# =================
# Guesses sized for a ~50 lb turbojet drone (GP solves don't need them, but
# IPOPT fallbacks and anyone reading the file do)
W_TO      = f.Variable(name="W_TO"     , guess=50.0  , units="lbf" , description="Takeoff gross weight")
W_ZF      = f.Variable(name="W_ZF"     , guess=42.0  , units="lbf" , description="Zero fuel weight")
W_empty   = f.Variable(name="W_empty"  , guess=37.0  , units="lbf" , description="Empty weight")
W_wing    = f.Variable(name="W_wing"   , guess=5.0   , units="lbf" , description="Wing weight")
W_eng     = f.Variable(name="W_eng"    , guess=18.0  , units="lbf" , description="Installed engine weight")
W_other   = f.Variable(name="W_other"  , guess=14.0  , units="lbf" , description="Fuselage, tails, fuel system, avionics")
W_f_climb = f.Variable(name="W_f_climb", guess=3.0   , units="lbf" , description="Takeoff/climb/accel fuel")
W_f_cruise= f.Variable(name="W_f_cruis", guess=0.2   , units="lbf" , description="Supersonic level-flight fuel")
W_f_res   = f.Variable(name="W_f_res"  , guess=2.0   , units="lbf" , description="Reserve + descent fuel")
W_cr_start= f.Variable(name="W_cr_strt", guess=47.0  , units="lbf" , description="Weight at start of cruise")
W_cr_end  = f.Variable(name="W_cr_end" , guess=44.0  , units="lbf" , description="Weight at end of cruise")
S         = f.Variable(name="S"        , guess=5.0   , units="ft^2", description="Wing reference area")
tau       = f.Variable(name="tau"      , guess=0.03  , units="-"   , description="Wing thickness to chord ratio")
C_L       = f.Variable(name="C_L"      , guess=0.03  , units="-"   , description="Cruise lift coefficient")
C_D       = f.Variable(name="C_D"      , guess=0.03  , units="-"   , description="Cruise drag coefficient")
C_D0      = f.Variable(name="C_D0"     , guess=0.012 , units="-"   , description="Skin friction drag coefficient")
C_Dw      = f.Variable(name="C_Dw"     , guess=0.012 , units="-"   , description="Wave drag coefficient")
D         = f.Variable(name="D"        , guess=60.0  , units="lbf" , description="Cruise drag")
T_SL      = f.Variable(name="T_SL"     , guess=200.0 , units="lbf" , description="Sea-level static thrust")
c_T       = f.Variable(name="c_T"      , guess=1.3   , units="1/hr", description="TSFC (engine-weight fit)")
c_sw      = f.Variable(name="c_sw"     , guess=0.5   , units="-"   , description="cos(wing leading-edge sweep)")
L_fus     = f.Variable(name="L_fus"    , guess=8.0   , units="ft"  , description="Fuselage length")
d_fus     = f.Variable(name="d_fus"    , guess=0.67  , units="ft"  , description="Fuselage diameter")
Swet_fus  = f.Variable(name="Swet_fus" , guess=17.0  , units="ft^2", description="Fuselage wetted area, pi d L")
CDA_fus_w = f.Variable(name="CDA_fus_w", guess=0.04  , units="ft^2", description="Fuselage wave drag area (Sears-Haack x E_WD)")
T_seg     = f.Variable(name="T_seg"    , guess=40.0  , units="lbf" , description="Thrust required in the supersonic segment")
h         = f.Variable(name="h"        , guess=40000.0, units="ft" , description="Cruise altitude")
rho       = f.Variable(name="rho"      , guess=5.9e-4, units="slug/ft^3", description="Air density at cruise altitude")
a         = f.Variable(name="a"        , guess=968.1 , units="ft/s", description="Speed of sound at cruise altitude")
lapse     = f.Variable(name="lapse"    , guess=0.35  , units="-"   , description="Cruise thrust / sea-level thrust")

# =================
# Declare Constants
# =================
# Mission requirements
H_FIXED   = float(sys.argv[1]) if len(sys.argv) > 1 else None    # optional: pin the cruise altitude (ft)
H_EXP     = 1.4916                         # exponent of the altitude fit h = h_fit sigma^-H_EXP (plain number)
M_cruise  = 1.0                            # plain number: drag model below needs it too
W_max     = f.Constant(name="W_max"    , value=55.0      , units="lbf"      , description="Max takeoff weight (Part 107 limit)")
W_pay     = f.Constant(name="W_pay"    , value=0.0       , units="lbf"      , description="Payload weight (sensor)")
t_super   = f.Constant(name="t_super"  , value=5.0       , units="s"        , description="Level supersonic flight time")
M         = f.Constant(name="M"        , value=M_cruise  , units="-"        , description="Cruise Mach number (drag held at M 1.2 values below 1.2)")
rho_SL    = f.Constant(name="rho_SL"   , value=2.377e-3  , units="slug/ft^3", description="Sea-level air density")
V_stall   = f.Constant(name="V_stall"  , value=120.0     , units="ft/s"     , description="Max stall speed (takeoff / landing)")
C_Lmax    = f.Constant(name="C_Lmax"   , value=1.0       , units="-"        , description="Max lift coefficient (small delta, no flaps)")
TW_TO     = f.Constant(name="TW_TO"    , value=0.5       , units="-"        , description="Required takeoff thrust/weight")

# Propulsion (model-scale turbojet, e.g. JetCat / KingTech class)
T_slope   = f.Constant(name="T_slope"  , value=7.1       , units="-"        , description="Engine fit: static thrust = T_slope W + T_int (W in lb)")
T_int     = f.Constant(name="T_int"    , value=28.6      , units="lbf"      , description="Engine fit thrust intercept")
tsfc_int  = f.Constant(name="tsfc_int" , value=1.617     , units="1/hr"     , description="Engine fit: TSFC = tsfc_int + tsfc_slope W")
tsfc_slope= f.Constant(name="tsfc_slope", value=-0.0117  , units="1/(hr*lbf)", description="Engine fit TSFC slope")
K_ram     = f.Constant(name="K_ram"    , value=0.5       , units="-"        , description="Ram recovery factor on thrust lapse (~1 near M 1)")
#research if what this factor is close to

# Aerodynamics
C_f       = f.Constant(name="C_f"      , value=0.003     , units="-"        , description="Skin friction coefficient (Re ~ 1e7)")
Swet_S    = f.Constant(name="Swet_S"   , value=2.05      , units="-"        , description="Wing wetted / reference area")
E_WD      = f.Constant(name="E_WD"     , value=1.8       , units="-"        , description="Wave drag efficiency factor (1.8 reproduces the old 6 ft x 8 in body, 0.086 ft^2)")
d_min     = f.Constant(name="d_min"    , value=8.0/12.0  , units="ft"       , description="Minimum fuselage diameter (engine)")
L_min     = f.Constant(name="L_min"    , value=4.0       , units="ft"       , description="Min fuselage length (placeholder)")
L_max     = f.Constant(name="L_max"    , value=12.0      , units="ft"       , description="Max fuselage length (placeholder)")
K_ind     = f.Constant(name="K_ind"    , value=1.2       , units="-"        , description="Lift-dependent drag factor multiplier")

# Structures / weights
AR        = f.Constant(name="AR"       , value=2.0       , units="-"        , description="Wing aspect ratio")
N_ult     = f.Constant(name="N_ult"    , value=6.0       , units="-"        , description="Ultimate load factor")
K_wing    = f.Constant(name="K_wing"   , value=0.0103    , units="-"        , description="Raymer wing coeff (1/cos(sweep) applied separately)")
tau_min   = f.Constant(name="tau_min"  , value=0.025     , units="-"        , description="Min t/c (fuel volume, structure)")
f_other   = f.Constant(name="f_other"  , value=0.20      , units="-"        , description="Other empty weight as fraction of W_TO")
W_fixed   = f.Constant(name="W_fixed"  , value=3.0       , units="lbf"      , description="Avionics, datalink, battery, servos")
f_TO      = f.Constant(name="f_TO"     , value=0.02      , units="-"        , description="Start/taxi/takeoff fuel fraction of W_TO")
V_climb   = f.Constant(name="V_climb"  , value=600.0     , units="ft/s"     , description="Average speed during climb/accel")
eta_climb = f.Constant(name="eta_clmb" , value=0.5       , units="-"        , description="Excess thrust fraction during climb, (T-D)/T")
g         = f.Constant(name="g"        , value=32.174    , units="ft/s^2"   , description="Gravitational acceleration")

# Standard atmosphere (troposphere), GP-compatible over 15,000-25,000 ft (+-2.7%)
h_fit     = f.Constant(name="h_fit"    , value=7720.7    , units="ft"       , description="Altitude fit h = h_fit*sigma^-1.4916 (+-2.7%)")
a_SL      = f.Constant(name="a_SL"     , value=1116.4    , units="ft/s"     , description="Sea-level speed of sound")
a_strat   = f.Constant(name="a_strat"  , value=968.1     , units="ft/s"     , description="Speed of sound above 36,089 ft")
sig_max   = f.Constant(name="sig_max"  , value=0.629     , units="-"        , description="Density ratio at 15,000 ft (lowest cruise)")
sig_min   = f.Constant(name="sig_min"  , value=0.4481    , units="-"        , description="Density ratio at 25,000 ft (ceiling of the fit)")
h_max     = f.Constant(name="h_max"    , value=20000.0   , units="ft"       , description="Max cruise altitude")
f_res     = f.Constant(name="f_res"    , value=0.06      , units="-"        , description="Reserve fuel as fraction of W_ZF")

h_ref     = f.Constant(name="h_ref"    , value=1.0       , units="ft"       , description="1 ft")

# Reference quantities so the imperial Raymer fit is dimensionally clean
lbf_ref   = f.Constant(name="lbf_ref"  , value=1.0       , units="lbf"      , description="1 lbf")
ft2_ref   = f.Constant(name="ft2_ref"  , value=1.0       , units="ft^2"     , description="1 ft^2")

# Derived constants (plain numbers, computed in python)
# Linear supersonic theory divides by sqrt(M^2 - 1), which is 0 at M = 1.
# Real wave drag peaks around M 1.05-1.2, so below M 1.2 hold the M 1.2
# values (Raymer's transonic approach); slightly conservative near M 1.
AR_val = 2.0                                       # keep equal to the AR constant above
AR_delta_c = AR_val / float(np.sqrt(16.0 + AR_val**2))   # = AR / sqrt(16 + AR^2) (plain number)
beta   = float(np.sqrt(max(M_cruise, 1.2)**2 - 1))
K_sup  = beta / 4                          # supersonic flat-plate C_Di = K C_L^2
K_wave = 4 / beta                          # thin-airfoil wave drag C_Dw = K tau^2

V     = M * a                              # cruise speed
sigma = rho / rho_SL                       # density ratio
q = 0.5 * rho * V**2                       # cruise dynamic pressure

# =====================
# Declare the Objective
# =====================
f.Objective(W_TO)

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
        # --- Weight build-up ---------------------------------------------
        W_TO    >= W_ZF + W_f_climb + W_f_cruise + W_f_res,
        W_ZF    >= W_pay + W_empty,
        W_empty >= W_wing + W_eng + W_other,
        W_other >= W_fixed + f_other * W_TO,

        # Wing weight, Raymer fighter/attack fit (lb, ft^2), GP monomial
        W_wing / lbf_ref >= K_wing * (N_ult * W_TO / lbf_ref)**0.5
                            * (S / ft2_ref)**0.622 * AR**0.785 * tau**-0.4 / c_sw,    #read the raymer equation what does this mean 

        # Engines sized by thrust
        # Engine fits (AMT / JetCat / PBS data). The offsets make these
        # signomial (not a pure GP), so the solve is no longer guaranteed global.
        W_eng >= (T_SL - T_int) / T_slope,                 # static thrust fit
        c_T   >= tsfc_int + tsfc_slope * W_eng,            # TSFC at max thrust fit

        # --- Mission fuel --------------------------------------------------
        # Energy method: fuel = c_T * thrust * time, and the work done by
        # excess thrust (T-D)*V*t must equal the energy gained W*(h + V^2/2g)
        W_f_climb  >= f_TO * W_TO
                      + c_T * W_TO * (h + V**2 / (2 * g)) / (V_climb * eta_climb),
        W_TO       >= W_cr_start + W_f_climb,
        W_f_res    >= f_res * W_ZF,
        W_cr_start >= W_cr_end + W_f_cruise,
        W_cr_end   >= W_ZF + W_f_res,

        # Power-based supersonic segment (level): T = D, fuel = c_T T t
        T_seg      >= D,
        W_f_cruise >= c_T * T_seg * t_super,

        # --- Cruise aerodynamics ------------------------------------------
        W_cr_start == q * C_L * S,  # lift = weight for steady level flight
        C_D  >= C_D0 + C_Dw + K_ind * K_sup * C_L**2,
        # Fuselage geometry: wetted area and Sears-Haack wave drag area
        #   CDA = E_WD (9 pi / 2)(A_max / L)^2,  A_max = pi d^2 / 4
        Swet_fus  >= np.pi * d_fus * L_fus,
        CDA_fus_w >= E_WD * 4.5 * np.pi * (np.pi / 4)**2 * d_fus**4 / L_fus**2,
        # Sweep: cos(sweep) <= 1; a delta of aspect ratio AR (AR = 4/tan(sweep))
        # sets the most sweep, i.e. c_sw >= AR / sqrt(16 + AR^2)
        c_sw <= 1.0,
        c_sw >= AR_delta_c,
        d_fus >= d_min,
        L_fus >= L_min,
        L_fus <= L_max,
        C_D0 >= C_f * Swet_S + C_f * Swet_fus / S,
        C_Dw >= K_wave * tau**2 * c_sw**2 + CDA_fus_w / S,   # sweep cuts wing wave drag by cos^2
        D    >= q * C_D * S,
        tau  >= tau_min,

        # --- Sizing requirements -------------------------------------------
        W_TO <= W_max,                                     # weight limit
        W_TO <= 0.5 * rho_SL * V_stall**2 * C_Lmax * S,   # landing / stall
        T_SL >= TW_TO * W_TO,                              # takeoff
        lapse * T_SL >= T_seg,                             # supersonic segment thrust

        # --- Altitude / atmosphere ------------------------------------------
        lapse == K_ram * sigma**0.7,                       # turbojet thrust lapse #verify formula this is available thrust at cruise
        h     >= h_fit * sigma**-H_EXP,                   # altitude from density
        h     <= h_max,                                    # cruise altitude ceiling
        sigma <= sig_max,
        sigma >= sig_min,
        a     >= a_SL * sigma**0.1175,                     # troposphere: a ~ T^0.5
        a     >= a_strat,                                  # stratosphere: a const
    ])

if H_FIXED is not None:                                    # altitude sweeps pin h
    f.ConstraintList([h == H_FIXED * h_ref])

# =====
# Solve
# =====
if __name__ == "__main__":
    # Engine fits are signomial, so let lcsolver pick the solver
    sol = lcsolver.solve(f)
    print(sol.summary())

    W = sol.variables()
    print(f"Fuel fraction   : {(W['W_f_climb'] + W['W_f_cruis'] + W['W_f_res']) / W['W_TO']:.3f}")
    print(f"Empty fraction  : {W['W_empty'] / W['W_TO']:.3f}")
    print(f"Cruise L/D      : {W['C_L'] / W['C_D']:.2f}")
