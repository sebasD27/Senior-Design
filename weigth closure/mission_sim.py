# ===========
# Description
# ===========
# Weight closure for a fuel-powered supersonic drone (<= 55 lb), formulated as a
# Geometric Program and solved with lcsolver.
#
# Objective: minimize takeoff gross weight W_TO.
#
# Mission:  takeoff / climb / accelerate  ->  5 s of level supersonic flight
#           (Breguet endurance)  ->  descent / landing with reserve fuel
#
# Models:
#   - Weight build-up: payload + empty weight + fuel, capped at W_max
#   - Empty weight: wing (Raymer fighter wing fit) + engines (engine T/W)
#     + everything else as a fraction of W_TO plus a fixed weight
#   - Supersonic drag: skin friction + wing wave drag (thin airfoil)
#     + fuselage wave drag (Sears-Haack area) + lift-dependent drag
#   - Supersonic fuel: Breguet endurance, exp() as a 4-term Taylor series
#     so the problem stays a GP (global optimum, no initial guess needed)
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
z_bre     = f.Variable(name="z_bre"    , guess=0.005 , units="-"   , description="Breguet exponent, ln(W_start/W_end)")
h         = f.Variable(name="h"        , guess=40000.0, units="ft" , description="Cruise altitude")
rho       = f.Variable(name="rho"      , guess=5.9e-4, units="slug/ft^3", description="Air density at cruise altitude")
a         = f.Variable(name="a"        , guess=968.1 , units="ft/s", description="Speed of sound at cruise altitude")
lapse     = f.Variable(name="lapse"    , guess=0.35  , units="-"   , description="Cruise thrust / sea-level thrust")

# =================
# Declare Constants
# =================
# Mission requirements
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
c_T       = f.Constant(name="c_T"      , value=1.5       , units="1/hr"     , description="Cruise TSFC")
K_ram     = f.Constant(name="K_ram"    , value=0.5       , units="-"        , description="Ram recovery factor on thrust lapse (~1 near M 1)")
#research if what this factor is close to
TW_eng    = f.Constant(name="TW_eng"   , value=9.0       , units="-"        , description="Installed engine thrust-to-weight")

# Aerodynamics
C_f       = f.Constant(name="C_f"      , value=0.003     , units="-"        , description="Skin friction coefficient (Re ~ 1e7)")
Swet_S    = f.Constant(name="Swet_S"   , value=2.05      , units="-"        , description="Wing wetted / reference area")
Swet_fus  = f.Constant(name="Swet_fus" , value=12.0      , units="ft^2"     , description="Fuselage + tail wetted area (6 ft x 8 in body)")
CDA_fus_w = f.Constant(name="CDA_fus_w", value=0.086     , units="ft^2"     , description="Fuselage wave drag area (Sears-Haack x E_WD)")
K_ind     = f.Constant(name="K_ind"    , value=1.2       , units="-"        , description="Lift-dependent drag factor multiplier")

# Structures / weights
AR        = f.Constant(name="AR"       , value=2.0       , units="-"        , description="Wing aspect ratio")
N_ult     = f.Constant(name="N_ult"    , value=6.0       , units="-"        , description="Ultimate load factor")
K_wing    = f.Constant(name="K_wing"   , value=0.0103*2.0, units="-"        , description="Raymer wing coeff x 1/cos(sweep 60 deg)")
tau_min   = f.Constant(name="tau_min"  , value=0.025     , units="-"        , description="Min t/c (fuel volume, structure)")
f_other   = f.Constant(name="f_other"  , value=0.20      , units="-"        , description="Other empty weight as fraction of W_TO")
W_fixed   = f.Constant(name="W_fixed"  , value=3.0       , units="lbf"      , description="Avionics, datalink, battery, servos")
f_TO      = f.Constant(name="f_TO"     , value=0.02      , units="-"        , description="Start/taxi/takeoff fuel fraction of W_TO")
V_climb   = f.Constant(name="V_climb"  , value=600.0     , units="ft/s"     , description="Average speed during climb/accel")
eta_climb = f.Constant(name="eta_clmb" , value=0.5       , units="-"        , description="Excess thrust fraction during climb, (T-D)/T")
g         = f.Constant(name="g"        , value=32.174    , units="ft/s^2"   , description="Gravitational acceleration")

# Standard atmosphere, GP-compatible over 25,000-55,000 ft
h_fit     = f.Constant(name="h_fit"    , value=17335.8   , units="ft"       , description="Altitude fit h = h_fit*sigma^-0.5701 (+-10%)")
a_SL      = f.Constant(name="a_SL"     , value=1116.4    , units="ft/s"     , description="Sea-level speed of sound")
a_strat   = f.Constant(name="a_strat"  , value=968.1     , units="ft/s"     , description="Speed of sound above 36,089 ft")
sig_max   = f.Constant(name="sig_max"  , value=0.4481    , units="-"        , description="Density ratio at 25,000 ft (lowest cruise)")
sig_min   = f.Constant(name="sig_min"  , value=0.1197    , units="-"        , description="Density ratio at 55,000 ft (ceiling)")
f_res     = f.Constant(name="f_res"    , value=0.06      , units="-"        , description="Reserve fuel as fraction of W_ZF")

# Reference quantities so the imperial Raymer fit is dimensionally clean
lbf_ref   = f.Constant(name="lbf_ref"  , value=1.0       , units="lbf"      , description="1 lbf")
ft2_ref   = f.Constant(name="ft2_ref"  , value=1.0       , units="ft^2"     , description="1 ft^2")

# Derived constants (plain numbers, computed in python)
# Linear supersonic theory divides by sqrt(M^2 - 1), which is 0 at M = 1.
# Real wave drag peaks around M 1.05-1.2, so below M 1.2 hold the M 1.2
# values (Raymer's transonic approach); slightly conservative near M 1.
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
                            * (S / ft2_ref)**0.622 * AR**0.785 * tau**-0.4,    #read the raymer equation what does this mean 

        # Engines sized by thrust
        W_eng >= T_SL / TW_eng,

        # --- Mission fuel --------------------------------------------------
        # Energy method: fuel = c_T * thrust * time, and the work done by
        # excess thrust (T-D)*V*t must equal the energy gained W*(h + V^2/2g)
        W_f_climb  >= f_TO * W_TO
                      + c_T * W_TO * (h + V**2 / (2 * g)) / (V_climb * eta_climb),
        W_TO       >= W_cr_start + W_f_climb,
        W_f_res    >= f_res * W_ZF,
        W_cr_start >= W_cr_end + W_f_cruise,
        W_cr_end   >= W_ZF + W_f_res,

        # Breguet endurance (constant L/D) for the timed supersonic leg:
        #   ln(W_start/W_end) = t c_T D / L
        z_bre >= t_super * c_T * C_D / C_L,
        W_f_cruise >= W_cr_end * (z_bre + z_bre**2/2 + z_bre**3/6 + z_bre**4/24),

        # --- Cruise aerodynamics ------------------------------------------
        W_cr_start == q * C_L * S,  #lift = weight for steady level flight
        C_D  >= C_D0 + C_Dw + K_ind * K_sup * C_L**2,
        C_D0 >= C_f * Swet_S + C_f * Swet_fus / S,
        C_Dw >= K_wave * tau**2 + CDA_fus_w / S,
        D    >= q * C_D * S,
        tau  >= tau_min,

        # --- Sizing requirements -------------------------------------------
        W_TO <= W_max,                                     # weight limit
        W_TO <= 0.5 * rho_SL * V_stall**2 * C_Lmax * S,   # landing / stall
        T_SL >= TW_TO * W_TO,                              # takeoff
        lapse * T_SL >= D,                                 # supersonic cruise

        # --- Altitude / atmosphere ------------------------------------------
        lapse == K_ram * sigma**0.7,                       # turbojet thrust lapse #verify formula this is available thrust at cruise
        h     >= h_fit * sigma**-0.5701,                   # altitude from density
        sigma <= sig_max,
        sigma >= sig_min,
        a     >= a_SL * sigma**0.1175,                     # troposphere: a ~ T^0.5
        a     >= a_strat,                                  # stratosphere: a const
    ])

# =====
# Solve
# =====
if __name__ == "__main__":
    # Force the GP solver: if the design can't close under W_max it raises
    # instead of falling back to raw IPOPT, which can return an infeasible
    # point while still reporting 'ok'
    sol = lcsolver.solve(f, solver='ipopt-convex')
    print(sol.summary())

    W = sol.variables()
    print(f"\nFuel fraction   : {(W['W_f_climb'] + W['W_f_cruis'] + W['W_f_res']) / W['W_TO']:.3f}")
    print(f"Empty fraction  : {W['W_empty'] / W['W_TO']:.3f}")
    print(f"Cruise L/D      : {W['C_L'] / W['C_D']:.2f}")
