"""Structural allowances affect both dry weight and dependent reserve fuel."""
from dataclasses import replace
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from missionsweep import MissionCase,Design,evaluate_one
from mission_atmosphere import density_ratio


def test_conservative_factors_increase_components_and_closure_consistently():
    nominal = MissionCase(engine_model="P1000-PRO",wing_weight_factor=1,fuselage_weight_factor=1)
    conservative = replace(nominal,wing_weight_factor=2,fuselage_weight_factor=1.5)
    design = Design(density_ratio(40000),.03,L_fus=8,d_fus=.8)
    before = evaluate_one(60,nominal,design)
    after = evaluate_one(60,conservative,design)
    assert after["W_wing"] == pytest.approx(2*before["W_wing"])
    assert after["W_fus"] == pytest.approx(1.5*before["W_fus"])
    extra = after["W_wing"]-before["W_wing"]+after["W_fus"]-before["W_fus"]
    assert after["W_empty"]-before["W_empty"] == pytest.approx(extra)
    assert after["W_f_res"]-before["W_f_res"] == pytest.approx(nominal.f_res*extra)
    assert after["D"] == pytest.approx(before["D"])


@pytest.mark.parametrize("factor", [0,.5,float('nan'),float('inf')])
def test_invalid_structural_allowances_are_rejected(factor):
    m = MissionCase(wing_weight_factor=factor)
    with pytest.raises(ValueError,match="structural weight factors"):
        evaluate_one(50,m,Design(density_ratio(40000),.03))


def test_quarter_chord_secant_matches_planform_geometry():
    import math
    from missionsweep import quarter_chord_secant
    for le in (0.0, 30.0, 63.4):
        quarter = math.atan(0.75 * math.tan(math.radians(le)))   # straight trailing edge
        assert quarter_chord_secant(math.cos(math.radians(le))) == pytest.approx(1 / math.cos(quarter))
    assert quarter_chord_secant(math.cos(math.radians(63.4))) < 1 / math.cos(math.radians(63.4))


def test_volume_check_counts_engine_fuel_and_systems():
    import math
    from missionsweep import MissionCase, fuselage_volume, required_volume
    m = MissionCase()
    assert fuselage_volume(m, 6.0, 0.5) == pytest.approx(m.f_vol_use * 3 * math.pi / 16 * math.pi / 4 * 0.25 * 6)
    base = required_volume(m, 0.3, 0.0)
    assert required_volume(m, 0.3, 5.0) - base == pytest.approx(m.k_tank * 5.0 / m.rho_fuel)
