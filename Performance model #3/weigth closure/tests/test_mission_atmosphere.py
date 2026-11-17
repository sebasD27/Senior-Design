"""Full-range atmosphere integration and outer altitude-search checks."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission_atmosphere import atmosphere_at_altitude, density_ratio, altitude_from_sigma, TROPOPAUSE_FT
from missionsweep import MissionCase, Design, evaluate_one, sigma_floor
from altitude_search import altitude_grid, run_altitude_search


@pytest.mark.parametrize("altitude", [0, 10000, TROPOPAUSE_FT, 40000, 50000])
def test_mission_uses_exact_atmosphere(altitude):
    mission = MissionCase(engine_model="P1000-PRO", lapse_override=0.179)
    result = evaluate_one(50, mission, Design(density_ratio(altitude), 0.025, L_fus=10, d_fus=mission.d_min))
    reference = atmosphere_at_altitude(altitude)
    assert result["h"] == pytest.approx(altitude, abs=1e-7)
    assert result["rho"] == pytest.approx(reference.density_slug_ft3, rel=1e-12)
    assert result["a"] == pytest.approx(reference.speed_of_sound_ft_s, rel=1e-12)
    assert result["lapse"] == 0.179


def test_ceiling_density_and_inverse_include_both_endpoints():
    assert density_ratio(0) == 1
    assert altitude_from_sigma(1) == 0
    assert altitude_from_sigma(density_ratio(50000)) == pytest.approx(50000)
    assert sigma_floor(MissionCase(h_max=40000)) == density_ratio(40000)


@pytest.mark.parametrize("bad", [-1, 50001, float('nan'), float('inf')])
def test_mission_range_is_enforced(bad):
    with pytest.raises(ValueError, match="0 and 50,000"):
        atmosphere_at_altitude(bad)


def test_altitude_search_keeps_exact_endpoints_and_tropopause():
    grid = altitude_grid(3000)
    assert grid[0] == 0
    assert grid[-1] == 50000
    assert TROPOPAUSE_FT in grid
    assert grid == sorted(set(grid))


def test_altitude_search_passes_hardware_flags_and_selects_best_sample(capsys):
    def solve(command, **kwargs):
        altitude = float(command[2])
        assert command[3:7] == ["--engine", "P1000-PRO", "--lapse", "0.179"]
        weight = 40 + ((altitude - 40000) / 10000)**2
        return SimpleNamespace(returncode=0, stdout=f"   W_TO : {weight}\n", stderr="")
    script = Path(__file__).resolve().parents[1] / "mission_sim.py"
    assert run_altitude_search(script, ["--engine", "P1000-PRO", "--lapse", "0.179"], 10000, run=solve) == 0
    assert "Best sampled altitude: 40000.0 ft" in capsys.readouterr().out


def test_altitude_search_reports_solver_failure(capsys):
    def fail(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="solver unavailable")
    assert run_altitude_search(Path("/tmp/mission_sim.py"), [], 50000, run=fail) == 1
    assert "No sampled altitude solved successfully" in capsys.readouterr().err


def test_altitude_search_passes_previous_checked_state(tmp_path,capsys):
    import json
    calls = []
    def solve(command, **kwargs):
        calls.append(command)
        if len(calls) > 1:
            previous = Path(command[command.index("--initial-state")+1])
            assert json.loads(previous.read_text())["variables"]["W_TO"] == 20
        output = Path(command[command.index("--state-output")+1])
        output.write_text(json.dumps({"variables":{"W_TO":20},"mode":"empirical"}))
        return SimpleNamespace(returncode=0,stdout="   W_TO : 20\n",stderr="")
    assert run_altitude_search(Path("/tmp/mission_sim.py"),[],50000,run=solve) == 0
    assert len(calls) == 3  # endpoints plus tropopause
    assert "checked local solution" in capsys.readouterr().out
