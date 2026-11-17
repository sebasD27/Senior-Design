"""Engine-fit validity checks without changing the aerodynamic model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from missionsweep import Design, MissionCase, evaluate_one, sigma_floor, size_drone
# Bookkeeping tests use the earlier, lossless-inlet propulsion so their reference designs still close.
LEGACY_THRUST = dict(K_ram=1.266656, lapse_exp=0.946451, eta_inst=1.0, d_clear=0.0)

from engine_choice import catalogue_engine, mission_lapse
from mission_atmosphere import density_ratio
import pytest


def test_out_of_catalogue_engine_cannot_report_convergence():
    mission = MissionCase(h_max=20000.0, **LEGACY_THRUST)
    result = size_drone(mission, Design(sigma_floor(mission), 0.025, d_fus=1.5))
    assert not result["converged"]
    assert not result["feasible"]
    assert not result["propulsion_valid"]
    assert "catalogue" in result["propulsion_error"]
    assert result["c_T"] > 0
    assert result["W_f_climb"] > 0


def test_inner_engine_iteration_conserves_fit_relationships():
    mission = MissionCase(h_max=40000.0, **LEGACY_THRUST)
    result = evaluate_one(30, mission, Design(sigma_floor(mission), 0.025, L_fus=12, d_fus=mission.d_min))
    assert result["propulsion_valid"]
    assert abs(result["c_T"] - (mission.tsfc_int + mission.tsfc_slope * result["W_eng"])) < 1e-10
    assert mission.W_eng_min <= result["W_eng"] <= mission.W_eng_max


def test_selected_engine_cannot_shrink_to_fit_minimum():
    mission = MissionCase(h_max=20000.0, engine_model="P1000-PRO", V_stall=120.0, **LEGACY_THRUST)
    engine = catalogue_engine(mission.engine_model)
    result = size_drone(mission, Design(sigma_floor(mission), 0.025, L_fus=10, d_fus=mission.d_min))
    assert result["converged"]
    assert result["W_eng"] == engine.weight
    assert result["T_SL"] == engine.thrust
    assert result["c_T"] == engine.tsfc
    assert result["d_fus"] >= engine.diameter
    assert result["W_TO"] > mission.W_max
    assert not result["feasible"]


def test_known_installation_and_payload_enter_weight_build_up_once():
    from dataclasses import replace
    mission = MissionCase(h_max=20000.0, engine_model="P1000-PRO", V_stall=120.0, **LEGACY_THRUST)
    design = Design(sigma_floor(mission), 0.025, L_fus=10, d_fus=mission.d_min)
    base = evaluate_one(50, mission, design)
    loaded = evaluate_one(50, replace(mission, W_prop_install=3, W_pay=2), design)
    assert loaded["W_empty"] - base["W_empty"] == pytest.approx(3)
    assert loaded["W_ZF"] - base["W_ZF"] == pytest.approx(5)
    assert loaded["W_TO_calc"] - base["W_TO_calc"] == pytest.approx(5 * (1 + mission.f_res + mission.f_land))


def test_undersized_catalogue_engine_is_rejected():
    mission = MissionCase(h_max=20000.0, engine_model="P220-PRO")
    result = size_drone(mission, Design(sigma_floor(mission), 0.025, L_fus=6, d_fus=1))
    assert not result["propulsion_valid"]
    assert not result["feasible"]
    assert "required thrust" in result["propulsion_error"]


def test_unusable_catalogue_entry_is_not_silently_fitted():
    with pytest.raises(ValueError, match="needs weight"):
        catalogue_engine("P850")


def test_user_lapse_is_total_thrust_ratio_not_density_coefficient():
    mission = MissionCase(h_max=20000.0, engine_model="P1000-PRO", lapse_override=0.179)
    design = Design(sigma_floor(mission), 0.025, L_fus=10, d_fus=mission.d_min)
    result = evaluate_one(50, mission, design)
    assert result["lapse"] == 0.179
    assert result["lapse"] * result["T_SL"] == pytest.approx(44.2667)
    assert not result["propulsion_valid"]
    assert result["T_SL_required"] >= result["D"] / 0.179
    assert result["T_SL_path"] > 0  # a lapse override must not bypass path checks


@pytest.mark.parametrize("override", [0, -0.179, float('nan'), float('inf')])
def test_invalid_lapse_cannot_enter_weight_closure(override):
    with pytest.raises(ValueError, match="finite and positive"):
        mission_lapse(0.53, 1.2608163, 0.9210521, override)


def test_acceleration_path_margin_sizes_engine_above_dash_thrust():
    from missionsweep import _evaluate
    design = Design(density_ratio(20000.0), 0.03, 1.0, 12.0, 0.5, 40.0)
    plain = _evaluate(40, MissionCase(accel_margin=0.0), design, 1.3, 0.5)
    margin = _evaluate(40, MissionCase(accel_margin=0.10), design, 1.3, 0.5)
    assert plain["T_SL_path"] > 0
    assert margin["T_SL_path"] == pytest.approx(1.10 * plain["T_SL_path"])
    assert margin["T_SL_path"] >= 1.10 * margin["T_seg"] / margin["lapse"] * (1 - 1e-9)
    assert margin["T_SL_required"] == pytest.approx(max(margin["T_SL_path"], margin["T_seg"] / margin["lapse"]))


def test_lapse_map_matches_sim_grid_points():
    from engine_choice import lapse_map
    assert lapse_map(0.0, 0.0) == pytest.approx(1.0)
    assert lapse_map(20000.0, 1.0) == pytest.approx(0.7068, abs=2e-3)


def test_idle_descent_and_landing_fuel_enter_the_closure():
    from dataclasses import replace
    mission = MissionCase(h_max=20000.0, engine_model="P1000-PRO", V_stall=120.0, **LEGACY_THRUST)
    design = Design(sigma_floor(mission), 0.025, L_fus=10, d_fus=mission.d_min)
    r = evaluate_one(50, mission, design)
    from gp_atmosphere import descent_fuel_factor
    assert 0.6 < descent_fuel_factor(r["h"]) < 0.75          # idle fuel flow falls with altitude
    assert r["W_f_desc"] == pytest.approx(mission.f_idle * r["c_T"] / 3600 * r["T_SL"] * r["h"]
                                          * descent_fuel_factor(r["h"]) / mission.RoD_desc)
    assert r["W_f_land"] == pytest.approx(mission.f_land * r["W_ZF"])
    glide = evaluate_one(50, replace(mission, f_idle=0.0), design)
    assert glide["W_f_desc"] == 0.0
    assert r["W_TO_calc"] - glide["W_TO_calc"] == pytest.approx(r["W_f_desc"])
