"""Verify physical balances of the installed optimizer against numerical code."""
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

lcsolver = pytest.importorskip("lcsolver")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission_sim import build_mission, solution_state, select_catalogue_engine, solve_mission, constraint_residual
from engine_choice import catalogue_engine
from mission_atmosphere import atmosphere_at_altitude, density_ratio
from mission_path import segment_loads
from missionsweep import Design, MissionCase, evaluate_one
from aero import drag_terms


@pytest.fixture(scope="module")
def solved():
    model = build_mission(40000, MissionCase(engine_model="P1000-PRO",W_max=80,L_min=6))
    sol = lcsolver.solve(model.formulation, sensitivities=False)
    return model, solution_state(model, sol)


def test_engine_and_atmosphere_cannot_shrink_or_drift(solved):
    model, r = solved
    assert model.mission.engine_model == "P1000-PRO"
    assert r["W_eng"] == pytest.approx(24.25)
    assert r["T_SL"] == pytest.approx(247.3)
    assert r["d_fus"] >= 234/304.8 - 1e-7
    assert r["W_eng"] < r["W_TO"] <= 80
    reference = atmosphere_at_altitude(40000)
    assert r["rho"] == reference.density_slug_ft3
    assert r["a"] == reference.speed_of_sound_ft_s
    assert not {"h", "rho", "a"} & model.variables.keys()


def test_optimizer_drag_is_the_shared_component_drag(solved):
    model, r = solved
    q = .5*r["rho"]*(model.mission.M*r["a"])**2
    terms, _ = drag_terms(model.mission,S=r["S"],tau=r["tau"],c_sw=r["c_sw"],
                          L=r["L_fus"],d=r["d_fus"],rho=r["rho"],
                          V=model.mission.M*r["a"],M=model.mission.M,C_L=r["C_L"])
    assert r["D"] == pytest.approx(q*r["S"]*sum(terms.values()),rel=1e-6)
    assert q*r["S"]*r["C_L"] == pytest.approx(r["W_cr_strt"],rel=1e-7)


def test_path_force_time_and_fuel_balances(solved):
    model, r = solved
    assert math.isfinite(r["path_fuel_check"])
    assert r["path_thrust_margin"] >= model.mission.accel_margin - 1e-6
    assert r["path_time"] == pytest.approx(r["path_time_check"],rel=1e-6)
    assert r["W_f_climb"] == pytest.approx(model.mission.f_TO*r["W_TO"]+r["path_fuel_check"],abs=1e-5)
    loads = segment_loads(model.mission,model.segments,weight=r["W_TO"],S=r["S"],
                          tau=r["tau"],c_sw=r["c_sw"],L=r["L_fus"],d=r["d_fus"])
    for i,(segment,drag,low,high) in enumerate(loads):
        available = low*r["T_SL"]
        assert available > drag
        assert r[f"dt_path_{i}"]*(available-drag) >= r["W_TO"]*segment.time_coefficient*(1-1e-6)
        assert r[f"dt_path_{i}"] >= segment.minimum_time - 1e-6


def test_numerical_mission_matches_the_solved_formulation(solved):
    model, r = solved
    m = model.mission
    area_factor = r["S"] / (2*r["W_TO"]/(m.rho_SL*m.V_stall**2*m.C_Lmax))
    d = Design(density_ratio(40000),r["tau"],area_factor,r["L_fus"],r["d_fus"],r["sweep_deg"])
    numeric = evaluate_one(r["W_TO"],m,d)
    assert numeric["propulsion_valid"]
    for key in ["D","W_eng","W_wing","W_fus","W_surf","W_f_climb","W_f_cruise","W_f_res"]:
        value = r["W_f_cruis"] if key == "W_f_cruise" else r[key]
        assert numeric[key] == pytest.approx(value,rel=1e-5,abs=1e-5)
    assert numeric["W_TO_calc"] == pytest.approx(r["W_TO"],abs=1e-5)


def test_explicit_cli_can_still_pin_p1000(tmp_path):
    script = Path(__file__).resolve().parents[1] / "mission_sim.py"
    result = subprocess.run([sys.executable,str(script),"40000","--engine","P1000-PRO"],capture_output=True,text=True,
                            env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/"matplotlib")),timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "Engine: P1000-PRO" in result.stdout
    assert "nan" not in result.stdout.lower()
    assert "inf lb" not in result.stdout
    assert "Path fuel numerical check:" in result.stdout


def test_measured_structural_floors_enter_optimization(solved):
    from dataclasses import replace
    model, _ = solved
    mission = replace(model.mission, W_wing_min=5, W_fus_min=8)
    loaded = build_mission(40000,mission)
    sol = lcsolver.solve(loaded.formulation,sensitivities=False)
    result = solution_state(loaded,sol)
    assert result["W_wing"] >= 5 - 1e-6
    assert result["W_fus"] >= 8 - 1e-6


def test_real_catalogue_selection_preserves_paired_specs(solved):
    model, _ = solved
    engines = [catalogue_engine("P1000-PRO"),catalogue_engine("P900-PRO-S-O-R")]
    selected = select_catalogue_engine(40000,model.mission,engines)
    assert selected.state is not None
    successful = [trial.weight for trial in selected.trials if trial.weight is not None]
    assert selected.state["W_TO"] == pytest.approx(min(successful))
    engine = catalogue_engine(selected.model.mission.engine_model)
    assert selected.state["W_eng"] == pytest.approx(engine.weight)
    assert selected.state["T_SL"] == pytest.approx(engine.thrust)
    assert selected.state["c_T"] == pytest.approx(engine.tsfc)
    assert selected.state["d_fus"] >= engine.diameter - 1e-6


def test_empirical_restart_avoids_spurious_30k_heavy_branch():
    mission = MissionCase(engine_model=None,W_max=80,L_min=6,wing_weight_factor=1,fuselage_weight_factor=1)
    model,sol,state = solve_mission(build_mission(30000,mission))
    # A feasible ~16 lb point is known; accepting the nominal-start ~74 lb
    # stationary point as the minimum is the regression this checks.
    assert state["W_TO"] < 25
    assert state["constraint_residual"] <= 1e-5
    successful = [weight for _,weight,_ in sol["mission_starts"] if weight is not None]
    assert state["W_TO"] == pytest.approx(min(successful))


def test_original_constraints_detect_invalid_solver_writeback(solved):
    model, _ = solved
    variable = model.variables["W_TO"]
    original = variable.value
    try:
        variable.set_value(1.0)
        assert constraint_residual(model) > 1e-5
    finally:
        variable.set_value(original)
