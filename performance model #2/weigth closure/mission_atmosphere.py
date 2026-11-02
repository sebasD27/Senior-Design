"""Shared exact standard atmosphere for mission altitudes from 0 to 50 kft.

The engine and mission use the same US Standard Atmosphere lower layers.
Public altitudes are geometric feet; layer equations use geopotential height.
"""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine sim" / "py-engine-sim" / "src"))
from enginesim.atmosphere import isa, altitude_from_density_ratio, geopotential_to_geometric, TROP_PAUSE_M
from enginesim.units import FT_TO_M

MIN_ALTITUDE_FT = 0.0
MAX_ALTITUDE_FT = 50000.0
TROPOPAUSE_FT = geopotential_to_geometric(TROP_PAUSE_M) / FT_TO_M
RHO_SL = isa(0).density_slug_ft3


def atmosphere_at_altitude(altitude_ft: float):
    if not math.isfinite(altitude_ft) or not MIN_ALTITUDE_FT <= altitude_ft <= MAX_ALTITUDE_FT:
        raise ValueError("mission altitude must be between 0 and 50,000 geometric ft")
    return isa(altitude_ft)


def density_ratio(altitude_ft: float) -> float:
    return atmosphere_at_altitude(altitude_ft).density_slug_ft3 / RHO_SL


def altitude_from_sigma(sigma: float) -> float:
    minimum = density_ratio(MAX_ALTITUDE_FT)
    if not math.isfinite(sigma) or not minimum * (1 - 1e-12) <= sigma <= 1.0 + 1e-12:
        raise ValueError("mission density ratio is outside the 0-50,000 ft atmosphere")
    return max(0.0, min(MAX_ALTITUDE_FT, altitude_from_density_ratio(max(minimum, min(1.0, sigma)))))
