from enginesim.presets import p1000_like
from enginesim.turbojet import FlightCondition, evaluate_turbojet
from enginesim.atmosphere import R_AIR
from enginesim.units import BTU_PER_LBM_TO_J_PER_KG, FT_LBF_PER_SLUG_R_TO_J_PER_KG_K, R_TO_K
import pytest


@pytest.mark.parametrize("altitude,mach,throttle", [(0, 0, 1), (20000, 1, 1), (30000, 1.5, 0.5), (0, 0, 0)])
def test_mass_thrust_and_energy_balances(altitude, mach, throttle):
    engine = p1000_like()
    result = evaluate_turbojet(engine, FlightCondition(altitude, mach), throttle)
    assert result.nozzle.mass_flow_lbm_s == pytest.approx(result.airflow_lbm_s + result.fuel_flow_lbm_s, rel=1e-12)
    assert result.gross_thrust_lbf == pytest.approx(result.nozzle.gross_thrust_lbf)
    assert result.pressure_thrust_lbf == pytest.approx((result.nozzle.exit_pressure_psf - result.ambient_pressure_psf) * result.nozzle.exit_area_in2 / 144)
    # 1 slug = 32.174... lbm in the imperial momentum equation.
    from enginesim.units import FT_TO_M, LBF_TO_N, LBM_TO_KG
    momentum = result.nozzle.mass_flow_lbm_s * LBM_TO_KG * result.nozzle.exit_velocity_ft_s * FT_TO_M / LBF_TO_N
    assert result.gross_thrust_lbf == pytest.approx(momentum + result.pressure_thrust_lbf)
    assert result.net_thrust_lbf == pytest.approx(result.gross_thrust_lbf - result.ram_drag_lbf)
    cpa = 1.4 * R_AIR / 0.4
    cpg = engine.gamma_hot * engine.gas_constant_hot_ft_lbf_slug_r * FT_LBF_PER_SLUG_R_TO_J_PER_KG_K / (engine.gamma_hot - 1)
    effective = engine.min_throttle + throttle * (engine.max_throttle - engine.min_throttle)
    t4 = (engine.idle_turbine_inlet_temperature_r + effective * (engine.turbine_inlet_temperature_r - engine.idle_turbine_inlet_temperature_r)) * R_TO_K
    heat_in = cpa * result.compressor_exit_temperature_r * R_TO_K + result.fuel_to_air_ratio * engine.burner_efficiency * engine.fuel_heating_value_btu_lbm * BTU_PER_LBM_TO_J_PER_KG
    assert heat_in == pytest.approx((1 + result.fuel_to_air_ratio) * cpg * t4, rel=1e-12)
    compressor_work = cpa * (result.compressor_exit_temperature_r - result.inlet_total_temperature_r) * R_TO_K
    turbine_work = (1 + result.fuel_to_air_ratio) * cpg * (t4 - result.turbine_exit_temperature_r * R_TO_K) * engine.mechanical_efficiency
    assert turbine_work == pytest.approx(compressor_work, rel=1e-12)


def test_preset_matches_reference_airflow_without_thrust_scaling():
    engine = p1000_like()
    result = evaluate_turbojet(engine, FlightCondition())
    assert result.airflow_lbm_s == pytest.approx(engine.design_airflow_lbm_s, rel=1e-8)


def test_p1000_like_returns_finite_result():
    result = evaluate_turbojet(p1000_like(), FlightCondition(0.0, 0.0))
    assert result.net_thrust_lbf > 0.0
    assert result.airflow_lbm_s > 0.0
    assert result.fuel_flow_lbm_s > 0.0
    assert result.net_thrust_lbf == result.gross_thrust_lbf - result.ram_drag_lbf


def test_ram_drag_increases_with_mach():
    engine = p1000_like()
    static = evaluate_turbojet(engine, FlightCondition(0.0, 0.0))
    fast = evaluate_turbojet(engine, FlightCondition(0.0, 0.8))
    assert fast.ram_drag_lbf > static.ram_drag_lbf
