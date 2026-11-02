"""Reference configurations used for demonstrations.

The P1000 preset is deliberately named ``p1000_like``: public JetCat data
does not expose the complete compressor/turbine/nozzle maps required for a
validated off-design model.
"""

from .turbojet import TurbojetConfig


def p1000_like() -> TurbojetConfig:
    return TurbojetConfig(
        name="P1000-like (screening model)",
        compressor_pressure_ratio=4.0,
        design_airflow_lbm_s=3.968,
        design_static_thrust_lbf=247.3,
        design_fuel_flow_lbm_s=2.9 * 0.80 * 2.20462262185 / 60.0,
        engine_mass_lbm=24.25,
        diameter_in=9.21,
        length_in=17.48,
        inlet_pressure_recovery=0.96,
        compressor_efficiency=0.78,
        burner_efficiency=0.98,
        burner_pressure_ratio=0.95,
        turbine_efficiency=0.84,
        mechanical_efficiency=0.98,
        turbine_inlet_temperature_r=2340.0,
        idle_turbine_inlet_temperature_r=1530.0,
        nozzle_efficiency=0.97,
        # Assumed effective throat, calibrated to 3.968 lbm/s SLS airflow
        # with the constant-property cycle. This is not a measured nozzle area.
        nozzle_throat_area_in2=12.3603693516,
        nozzle_exit_area_ratio=1.0,
    )
