from enginesim.atmosphere import isa
from enginesim.atmosphere import altitude_from_density_ratio, geopotential_to_geometric
from enginesim.units import FT_TO_M, PSF_TO_PA, R_TO_K, SLUG_TO_KG
import pytest


def test_sea_level_standard_atmosphere():
    state = isa(0.0)
    assert abs(state.temperature_r - 518.67) < 1e-8
    assert abs(state.pressure_psf - 2116.2166) < 0.01
    assert abs(state.density_slug_ft3 - 0.0023769) < 1e-6


def test_atmosphere_decreases_with_altitude():
    low = isa(0.0)
    high = isa(3000.0)
    assert high.pressure_psf < low.pressure_psf
    assert high.density_slug_ft3 < low.density_slug_ft3


@pytest.mark.parametrize("height_m,temperature_k,pressure_pa,density_kg_m3", [
    (0, 288.15, 101325.0, 1.225000),
    (10000, 223.15, 26436.3, 0.412707),
    (11000, 216.65, 22632.1, 0.363918),
    (15000, 216.65, 12044.6, 0.193674),
])
def test_standard_atmosphere_reference_levels(height_m, temperature_k, pressure_pa, density_kg_m3):
    # US Standard Atmosphere 1976 levels are geopotential, not geometric.
    state = isa(geopotential_to_geometric(height_m) / FT_TO_M)
    assert state.temperature_r * R_TO_K == pytest.approx(temperature_k, abs=1e-8)
    assert state.pressure_psf * PSF_TO_PA == pytest.approx(pressure_pa, rel=5e-6)
    assert state.density_slug_ft3 * SLUG_TO_KG / FT_TO_M**3 == pytest.approx(density_kg_m3, rel=5e-6)


def test_dense_zero_to_50k_density_inverse_and_monotonicity():
    rho0 = isa(0).density_slug_ft3
    previous = isa(0)
    for altitude in range(0, 50001, 50):
        state = isa(altitude)
        recovered = altitude_from_density_ratio(state.density_slug_ft3 / rho0)
        assert recovered == pytest.approx(altitude, abs=1e-7)
        assert state.pressure_psf <= previous.pressure_psf
        assert state.density_slug_ft3 <= previous.density_slug_ft3
        previous = state


def test_tropopause_is_continuous_and_stratosphere_is_isothermal():
    altitude = geopotential_to_geometric(11000) / FT_TO_M
    below, above = isa(altitude - 1e-3), isa(altitude + 1e-3)
    assert below.temperature_r == pytest.approx(above.temperature_r, rel=1e-8)
    assert below.pressure_psf == pytest.approx(above.pressure_psf, rel=1e-6)
    assert below.density_slug_ft3 == pytest.approx(above.density_slug_ft3, rel=1e-6)
    assert isa(40000).speed_of_sound_ft_s == pytest.approx(isa(50000).speed_of_sound_ft_s)


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -20000, 70000])
def test_bad_altitude_is_rejected(bad):
    with pytest.raises(ValueError):
        isa(bad)
