"""International Standard Atmosphere helpers using imperial API units."""

from dataclasses import dataclass
import math

from .units import FT_TO_M, K_TO_R, PSF_TO_PA, R_TO_K, SLUG_TO_KG, r_to_f


G0 = 9.80665
R_AIR = 287.05287
GAMMA_AIR = 1.4
T_SEA_LEVEL = 288.15
P_SEA_LEVEL = 101325.0
LAPSE = -0.0065
TROP_PAUSE_M = 11000.0
EARTH_RADIUS_M = 6356766.0  # US Standard Atmosphere 1976 geopotential radius


def geopotential_to_geometric(altitude_m: float) -> float:
    return EARTH_RADIUS_M * altitude_m / (EARTH_RADIUS_M - altitude_m)


def geometric_to_geopotential(altitude_m: float) -> float:
    return EARTH_RADIUS_M * altitude_m / (EARTH_RADIUS_M + altitude_m)


@dataclass(frozen=True)
class Atmosphere:
    altitude_ft: float
    temperature_r: float
    pressure_psf: float
    density_slug_ft3: float
    speed_of_sound_ft_s: float
    gamma: float = GAMMA_AIR
    gas_constant_ft_lbf_slug_r: float = 1716.59

    @property
    def temperature_f(self) -> float:
        """Temperature in degrees Fahrenheit, for display convenience."""
        return r_to_f(self.temperature_r)


def _isa_si(altitude_m: float) -> tuple[float, float, float, float]:
    """Return temperature, pressure, density, and sound speed in SI units."""
    if not math.isfinite(altitude_m) or altitude_m < -5000.0 or altitude_m > 20000.0:
        raise ValueError("isa() supports altitudes from -16404 ft through 65617 ft")

    h = geometric_to_geopotential(float(altitude_m))
    if h <= TROP_PAUSE_M:
        temperature = T_SEA_LEVEL + LAPSE * h
        pressure = P_SEA_LEVEL * (temperature / T_SEA_LEVEL) ** (
            -G0 / (LAPSE * R_AIR)
        )
    else:
        temperature = T_SEA_LEVEL + LAPSE * TROP_PAUSE_M
        pressure_tropopause = P_SEA_LEVEL * (temperature / T_SEA_LEVEL) ** (
            -G0 / (LAPSE * R_AIR)
        )
        pressure = pressure_tropopause * math.exp(
            -G0 * (h - TROP_PAUSE_M) / (R_AIR * temperature)
        )
    density = pressure / (R_AIR * temperature)
    sound = math.sqrt(GAMMA_AIR * R_AIR * temperature)
    return temperature, pressure, density, sound


def altitude_from_density_ratio(sigma: float) -> float:
    """Invert ISA density analytically; return geometric altitude in feet."""
    rho_sl = P_SEA_LEVEL / (R_AIR * T_SEA_LEVEL)
    sigma_low = _isa_si(20000.0)[2] / rho_sl
    sigma_high = _isa_si(-5000.0)[2] / rho_sl
    if not math.isfinite(sigma) or not sigma_low <= sigma <= sigma_high:
        raise ValueError("density ratio is outside the supported ISA altitude range")
    t_trop = T_SEA_LEVEL + LAPSE * TROP_PAUSE_M
    exponent = -G0 / (LAPSE * R_AIR) - 1.0
    sigma_trop = (t_trop / T_SEA_LEVEL)**exponent
    if sigma >= sigma_trop:
        h = T_SEA_LEVEL / LAPSE * math.expm1(math.log(sigma) / exponent)
    else:
        h = TROP_PAUSE_M - R_AIR * t_trop / G0 * math.log(sigma / sigma_trop)
    return geopotential_to_geometric(h) / FT_TO_M


def isa(altitude_ft: float) -> Atmosphere:
    """Return a standard-atmosphere state for geometric altitude in feet."""
    altitude_m = float(altitude_ft) * FT_TO_M
    temperature_k, pressure_pa, density_kg_m3, sound_m_s = _isa_si(altitude_m)
    return Atmosphere(
        altitude_ft=float(altitude_ft),
        temperature_r=temperature_k * K_TO_R,
        pressure_psf=pressure_pa / PSF_TO_PA,
        density_slug_ft3=density_kg_m3 * FT_TO_M**3 / SLUG_TO_KG,
        speed_of_sound_ft_s=sound_m_s / FT_TO_M,
    )


def stagnation_temperature(static_temperature_r: float, mach: float, gamma: float = GAMMA_AIR) -> float:
    return static_temperature_r * (1.0 + 0.5 * (gamma - 1.0) * mach**2)


def stagnation_pressure(static_pressure_psf: float, mach: float, gamma: float = GAMMA_AIR) -> float:
    return static_pressure_psf * (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (gamma / (gamma - 1.0))
