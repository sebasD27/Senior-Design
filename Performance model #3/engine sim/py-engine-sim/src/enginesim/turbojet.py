"""Reduced-order, single-spool dry turbojet model with an imperial API."""

from dataclasses import dataclass, asdict
import math
from typing import Any

from .atmosphere import R_AIR, isa
from .nozzle import NozzleResult, solve_nozzle
from .units import (
    BTU_PER_LBM_TO_J_PER_KG,
    FT_LBF_PER_SLUG_R_TO_J_PER_KG_K,
    FT_TO_M,
    K_TO_R,
    LBM_TO_KG,
    LBF_TO_N,
    PSF_TO_PA,
    R_TO_K,
)


@dataclass(frozen=True)
class FlightCondition:
    altitude_ft: float = 0.0
    mach: float = 0.0


@dataclass(frozen=True)
class TurbojetConfig:
    name: str = "generic-turbojet"
    compressor_pressure_ratio: float = 4.0
    design_airflow_lbm_s: float = 3.968
    design_static_thrust_lbf: float | None = None
    design_fuel_flow_lbm_s: float | None = None
    engine_mass_lbm: float | None = None
    diameter_in: float | None = None
    length_in: float | None = None
    inlet_pressure_recovery: float = 0.85
    compressor_efficiency: float = 0.78
    burner_efficiency: float = 0.98
    burner_pressure_ratio: float = 0.95
    turbine_efficiency: float = 0.84
    mechanical_efficiency: float = 0.98
    turbine_inlet_temperature_r: float = 2340.0
    idle_turbine_inlet_temperature_r: float = 1530.0
    nozzle_efficiency: float = 0.97
    nozzle_throat_area_in2: float = 23.25
    nozzle_exit_area_ratio: float = 1.0
    gamma_hot: float = 1.30
    gas_constant_hot_ft_lbf_slug_r: float = 1716.59
    fuel_heating_value_btu_lbm: float = 18475.0
    # Legacy empirical-airflow settings, retained for constructor compatibility.
    # Flow is now set by fixed-nozzle capacity, not this independent flow fit.
    reference_pressure_psf: float = 2116.2166
    reference_temperature_r: float = 518.67
    reference_airflow_lbm_s: float = 3.968
    airflow_exponent: float = 0.85
    compressor_face_mach: float = 0.20
    max_throttle: float = 1.0
    min_throttle: float = 0.20


@dataclass(frozen=True)
class TurbojetResult:
    name: str
    altitude_ft: float
    mach: float
    ambient_temperature_r: float
    ambient_pressure_psf: float
    ambient_density_slug_ft3: float
    flight_speed_ft_s: float
    inlet_total_pressure_psf: float
    inlet_total_temperature_r: float
    compressor_exit_pressure_psf: float
    compressor_exit_temperature_r: float
    burner_exit_pressure_psf: float
    turbine_exit_pressure_psf: float
    turbine_exit_temperature_r: float
    airflow_lbm_s: float
    fuel_flow_lbm_s: float
    fuel_to_air_ratio: float
    gross_thrust_lbf: float
    ram_drag_lbf: float
    pressure_thrust_lbf: float
    net_thrust_lbf: float
    specific_fuel_consumption_lbm_lbf_h: float
    nozzle: NozzleResult

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nozzle"] = asdict(self.nozzle)
        return data


def evaluate_turbojet(
    config: TurbojetConfig,
    flight: FlightCondition,
    throttle: float = 1.0,
) -> TurbojetResult:
    """Evaluate a reduced-order dry turbojet at one imperial-unit condition."""
    if not 0.0 <= throttle <= 1.0:
        raise ValueError("throttle must be between 0 and 1")
    if not math.isfinite(flight.mach) or flight.mach < 0.0:
        raise ValueError("Mach number cannot be negative")
    if not 0.0 < config.inlet_pressure_recovery <= 1.0:
        raise ValueError("inlet pressure recovery must be in (0, 1]")
    for name in ("compressor_efficiency", "burner_efficiency", "burner_pressure_ratio", "turbine_efficiency", "mechanical_efficiency"):
        if not 0.0 < getattr(config, name) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if not math.isfinite(config.gamma_hot) or config.gamma_hot <= 1.0 or not math.isfinite(config.gas_constant_hot_ft_lbf_slug_r) or config.gas_constant_hot_ft_lbf_slug_r <= 0:
        raise ValueError("hot-gas gamma must exceed one and gas constant must be positive")
    if not math.isfinite(config.compressor_pressure_ratio) or config.compressor_pressure_ratio < 1.0:
        raise ValueError("compressor pressure ratio must be at least one")
    if not 0 <= config.min_throttle <= config.max_throttle <= 1:
        raise ValueError("throttle limits must satisfy 0 <= min <= max <= 1")

    atm = isa(flight.altitude_ft)
    gamma = atm.gamma
    ambient_temperature_k = atm.temperature_r * R_TO_K
    ambient_pressure_pa = atm.pressure_psf * PSF_TO_PA
    speed = flight.mach * atm.speed_of_sound_ft_s * FT_TO_M
    t_t0 = ambient_temperature_k * (1.0 + 0.5 * (gamma - 1.0) * flight.mach**2)
    p_t0 = ambient_pressure_pa * (1.0 + 0.5 * (gamma - 1.0) * flight.mach**2) ** (
        gamma / (gamma - 1.0)
    )
    p_t2 = p_t0 * config.inlet_pressure_recovery
    t_t2 = t_t0

    effective_throttle = config.min_throttle + throttle * (
        config.max_throttle - config.min_throttle
    )
    # Calorically perfect gases: cp, gamma and R must describe the same gas.
    cpa = gamma * R_AIR / (gamma - 1.0)
    r_hot = config.gas_constant_hot_ft_lbf_slug_r * FT_LBF_PER_SLUG_R_TO_J_PER_KG_K
    cpg = config.gamma_hot * r_hot / (config.gamma_hot - 1.0)
    rcp = (config.compressor_pressure_ratio - 1.0) * effective_throttle + 1.0
    t_t3s = t_t2 * rcp ** ((gamma - 1.0) / gamma)
    t_t3 = t_t2 + (t_t3s - t_t2) / config.compressor_efficiency
    p_t3 = p_t2 * rcp

    t_t4 = (
        config.idle_turbine_inlet_temperature_r * R_TO_K
        + effective_throttle
        * (
            config.turbine_inlet_temperature_r * R_TO_K
            - config.idle_turbine_inlet_temperature_r * R_TO_K
        )
    )
    if not math.isfinite(t_t4) or t_t4 <= 0:
        raise ValueError("burner temperature must be finite and positive")
    p_t4 = p_t3 * config.burner_pressure_ratio
    available_heat_per_kg_fuel = (
        config.burner_efficiency * config.fuel_heating_value_btu_lbm * BTU_PER_LBM_TO_J_PER_KG
        - cpg * t_t4
    )
    required_heat_per_kg_air = cpg * t_t4 - cpa * t_t3
    if t_t4 <= t_t3 or required_heat_per_kg_air <= 0 or available_heat_per_kg_fuel <= 0:
        raise ValueError("specified burner temperature/heating value cannot sustain a fired cycle")
    fuel_to_air = required_heat_per_kg_air / available_heat_per_kg_fuel

    # Specific shaft work avoids an arbitrary flow floor. Nozzle capacity below
    # sets the common flow through the cycle; no compressor map is available.
    compressor_work = cpa * (t_t3 - t_t2)
    turbine_drop = compressor_work / ((1.0 + fuel_to_air) * cpg * config.mechanical_efficiency)
    t_t5 = t_t4 - turbine_drop
    t_t5s = t_t4 - turbine_drop / config.turbine_efficiency
    if t_t5s <= 0:
        raise ValueError("turbine cannot supply the requested compressor work")
    turbine_pressure_ratio = (t_t5s / t_t4) ** (
        config.gamma_hot / (config.gamma_hot - 1.0)
    )
    p_t5 = p_t4 * turbine_pressure_ratio

    nozzle = solve_nozzle(
        total_pressure_psf=p_t5 / PSF_TO_PA,
        total_temperature_r=t_t5 * K_TO_R,
        ambient_pressure_psf=atm.pressure_psf,
        throat_area_in2=config.nozzle_throat_area_in2,
        exit_area_ratio=config.nozzle_exit_area_ratio,
        gamma=config.gamma_hot,
        gas_constant_ft_lbf_slug_r=config.gas_constant_hot_ft_lbf_slug_r,
        nozzle_efficiency=config.nozzle_efficiency,
        flight_speed_ft_s=speed / FT_TO_M,
    )
    nozzle_mass_flow_kg_s = nozzle.mass_flow_lbm_s * LBM_TO_KG
    airflow = nozzle_mass_flow_kg_s / (1.0 + fuel_to_air)
    fuel_flow = airflow * fuel_to_air
    gross_n = nozzle.gross_thrust_lbf * LBF_TO_N
    ram_drag_n = airflow * speed
    net_n = gross_n - ram_drag_n
    gross_lbf = gross_n / LBF_TO_N
    ram_drag_lbf = ram_drag_n / LBF_TO_N
    net_lbf = net_n / LBF_TO_N
    fuel_flow_lbm_s = fuel_flow / LBM_TO_KG
    sfc = fuel_flow_lbm_s * 3600.0 / net_lbf if net_lbf > 1e-9 else math.inf

    return TurbojetResult(
        name=config.name,
        altitude_ft=flight.altitude_ft,
        mach=flight.mach,
        ambient_temperature_r=atm.temperature_r,
        ambient_pressure_psf=atm.pressure_psf,
        ambient_density_slug_ft3=atm.density_slug_ft3,
        flight_speed_ft_s=speed / FT_TO_M,
        inlet_total_pressure_psf=p_t2 / PSF_TO_PA,
        inlet_total_temperature_r=t_t2 * K_TO_R,
        compressor_exit_pressure_psf=p_t3 / PSF_TO_PA,
        compressor_exit_temperature_r=t_t3 * K_TO_R,
        burner_exit_pressure_psf=p_t4 / PSF_TO_PA,
        turbine_exit_pressure_psf=p_t5 / PSF_TO_PA,
        turbine_exit_temperature_r=t_t5 * K_TO_R,
        airflow_lbm_s=airflow / LBM_TO_KG,
        fuel_flow_lbm_s=fuel_flow_lbm_s,
        fuel_to_air_ratio=fuel_to_air,
        gross_thrust_lbf=gross_lbf,
        ram_drag_lbf=ram_drag_lbf,
        pressure_thrust_lbf=nozzle.pressure_thrust_lbf,
        net_thrust_lbf=net_lbf,
        specific_fuel_consumption_lbm_lbf_h=sfc,
        nozzle=nozzle,
    )
