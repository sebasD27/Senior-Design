"""The GP troposphere relations are the ISA, and the lapse fits stay close to the lapse they replace."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gp_atmosphere as gpa
from mission_atmosphere import atmosphere_at_altitude, RHO_SL
from mission_path import thrust_lapse
from missionsweep import MissionCase


@pytest.mark.parametrize("h_gp", [0.0, 5000.0, 20000.0, 36000.0])
def test_troposphere_relations_are_exact_isa(h_gp):
    theta = 1 - h_gp / gpa.H_T
    state = atmosphere_at_altitude(gpa.geometric_ft(h_gp))
    assert RHO_SL * theta**gpa.SIGMA_EXP == pytest.approx(state.density_slug_ft3, rel=1e-12)
    assert gpa.A_SL * theta**0.5 == pytest.approx(state.speed_of_sound_ft_s, rel=1e-12)


@pytest.mark.parametrize("mach", [0.65, 0.8, 1.0])
def test_fixed_mach_lapse_fit_error_is_small(mach):
    m = MissionCase()
    _, _, error = gpa.fit_lapse_power(lambda h: thrust_lapse(m, h, mach))
    assert error < 0.03
