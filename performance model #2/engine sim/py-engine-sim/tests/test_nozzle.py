from enginesim.nozzle import solve_nozzle
from enginesim.nozzle import _area_ratio, _mach_for_area_ratio
import pytest


def test_nozzle_chokes_and_produces_positive_thrust():
    result = solve_nozzle(
        total_pressure_psf=5220.0,
        total_temperature_r=1800.0,
        ambient_pressure_psf=2116.2,
        throat_area_in2=15.5,
    )
    assert result.choked
    assert result.exit_mach == 1.0
    assert result.mass_flow_lbm_s > 0.0
    assert result.gross_thrust_lbf > 0.0


def test_nozzle_expansion_increases_exit_mach():
    result = solve_nozzle(
        total_pressure_psf=10440.0,
        total_temperature_r=2160.0,
        ambient_pressure_psf=2116.2,
        throat_area_in2=15.5,
        exit_area_ratio=2.0,
    )
    assert result.choked
    assert result.exit_mach > 1.0


@pytest.mark.parametrize("supersonic", [True, False])
@pytest.mark.parametrize("area_ratio", [1.01, 2.0, 10.0])
def test_area_mach_round_trip(area_ratio, supersonic):
    mach = _mach_for_area_ratio(area_ratio, 1.3, supersonic)
    assert _area_ratio(mach, 1.3) == pytest.approx(area_ratio, rel=1e-12)
    assert (mach > 1) == supersonic


def test_cd_nozzle_subsonic_exit_uses_exit_area():
    convergent = solve_nozzle(2200, 1800, 2116.2, 15.5)
    divergent = solve_nozzle(2200, 1800, 2116.2, 15.5, exit_area_ratio=2)
    assert not divergent.choked
    assert divergent.exit_area_in2 == pytest.approx(31.0)
    assert divergent.mass_flow_lbm_s == pytest.approx(2 * convergent.mass_flow_lbm_s)


def test_internal_shock_is_rejected_instead_of_using_wrong_branch():
    with pytest.raises(ValueError, match="internal nozzle shock"):
        solve_nozzle(4000, 1800, 3000, 15.5, exit_area_ratio=2)


def test_zero_pressure_drop_has_zero_flow_and_thrust():
    result = solve_nozzle(2116.2, 1800, 2116.2, 15.5)
    assert result.mass_flow_lbm_s == pytest.approx(0, abs=1e-12)
    assert result.gross_thrust_lbf == pytest.approx(0, abs=1e-12)


def test_inconsistent_incoming_flow_is_rejected():
    with pytest.raises(ValueError, match="does not match nozzle capacity"):
        solve_nozzle(5220, 1800, 2116.2, 15.5, incoming_mass_flow_lbm_s=0.01)
