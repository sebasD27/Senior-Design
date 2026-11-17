"""ISA troposphere in GP form, so the dash altitude can be an optimization variable.

With theta = T / T_SL and h the geopotential altitude in ft (US Standard Atmosphere 1976):
    theta = 1 - h / H_T,   sigma = theta^SIGMA_EXP,   a = A_SL theta^0.5
These are exact in the troposphere (0 - 36,089 ft geopotential). theta = 1 - h/H_T is a difference,
so it enters the GP as two inequalities: theta + h/H_T <= 1 (posynomial) and h + H_T theta >= H_T
(signomial, handled by lcsolver's local solve). The stratosphere is isothermal, with density falling
exponentially, which has no GP form, so the free-altitude model stops at the tropopause.

Thrust lapse at a fixed Mach (or fixed speed) is fitted as K sigma^p over the troposphere from the
same lapse function the fixed-altitude model uses (mission_path.thrust_lapse).
"""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine sim" / "py-engine-sim" / "src"))
from enginesim.atmosphere import (EARTH_RADIUS_M, GAMMA_AIR, LAPSE, R_AIR, T_SEA_LEVEL, TROP_PAUSE_M,
                                  geopotential_to_geometric, isa)
from enginesim.units import FT_TO_M

G0 = 9.80665
H_T = T_SEA_LEVEL / -LAPSE / FT_TO_M                    # ft, geopotential height where theta would reach 0
SIGMA_EXP = G0 / (R_AIR * -LAPSE) - 1.0                 # 4.2559
A_SL = math.sqrt(GAMMA_AIR * R_AIR * T_SEA_LEVEL) / FT_TO_M   # ft/s
H_TROPO_GP = TROP_PAUSE_M / FT_TO_M                     # ft geopotential (36,089)
H_TROPO_GEOM = geopotential_to_geometric(TROP_PAUSE_M) / FT_TO_M
A_TROPO = A_SL * math.sqrt(1.0 - H_TROPO_GP / H_T)


def geometric_ft(h_geopotential_ft):
    """Geometric altitude (what mission_atmosphere and the fixed-altitude model use)."""
    r = EARTH_RADIUS_M / FT_TO_M
    return r * h_geopotential_ft / (r - h_geopotential_ft)


def geopotential_ft(h_geometric_ft):
    """Inverse of geometric_ft."""
    r = EARTH_RADIUS_M / FT_TO_M
    return r * h_geometric_ft / (r + h_geometric_ft)


def fit_lapse_power(lapse_of_altitude, samples=73):
    """Fit lapse = K sigma^p over the troposphere. lapse_of_altitude takes geometric ft.
    Returns K, p and the largest relative error of the fit."""
    heights = np.linspace(0.0, H_TROPO_GEOM, samples)
    sigma = np.array([isa(h).density_slug_ft3 for h in heights]) / isa(0.0).density_slug_ft3
    lapse = np.array([lapse_of_altitude(float(h)) for h in heights])
    p, log_k = np.polyfit(np.log(sigma), np.log(lapse), 1)
    k = math.exp(log_k)
    return k, float(p), float(np.max(np.abs(k * sigma ** p / lapse - 1.0)))


# Idle descent fuel. At a fixed engine setting the fuel flow scales with inlet conditions as
# delta sqrt(theta) (corrected-flow similarity), so idle fuel flow falls with altitude. For a descent at
# constant rate from h to the ground, fuel = f_idle x (SLS full-thrust fuel flow) x (h / RoD) x the mean
# of delta sqrt(theta) over the descent. In the troposphere delta sqrt(theta) = theta^IDLE_EXP and
#   h x mean = H_T (1 - theta_h^(IDLE_EXP + 1)) / (IDLE_EXP + 1)          (exact; used by the free GP)
IDLE_EXP = SIGMA_EXP + 1.5


def descent_fuel_factor(h_geometric_ft, samples=401):
    """Mean of delta sqrt(theta) from the ground to h (any altitude in the 0-50 kft ISA; ram ignored)."""
    if h_geometric_ft <= 0:
        return 1.0
    sl = isa(0.0)
    z = np.linspace(0.0, h_geometric_ft, samples)
    f = np.array([isa(x).pressure_psf / sl.pressure_psf * math.sqrt(isa(x).temperature_r / sl.temperature_r)
                  for x in z])
    return float(np.sum((f[1:] + f[:-1]) / 2) / (samples - 1))
