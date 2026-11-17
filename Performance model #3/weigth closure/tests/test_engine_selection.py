"""Catalogue completeness and aircraft-weight ranking for engine selection."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine_choice import catalogue_candidates, catalogue_engine
from missionsweep import MissionCase
import mission_sim


def test_catalogue_candidates_keep_complete_engine_specs_together():
    candidates, skipped = catalogue_candidates()
    assert candidates
    assert catalogue_engine("P1000-PRO") in candidates
    assert all(e.weight > 0 and e.thrust > 0 and e.tsfc > 0 and e.diameter > 0 for e in candidates)
    assert len({e.name for e in candidates}) == len(candidates)
    skipped_names = {name for name, reason in skipped}
    assert "JetCat P850" in skipped_names
    assert "JetCat P250-PRO-S" in skipped_names  # no diameter in the current CSV


def test_selection_minimizes_aircraft_weight_not_engine_weight(monkeypatch):
    light = catalogue_engine("P190-PRO-S")
    heavy = catalogue_engine("P1000-PRO")
    failed = catalogue_engine("Olympus HP")
    original = MissionCase(engine_model="auto")
    def solve(altitude, mission, engine):
        assert altitude == 40000
        assert mission is original
        if engine == failed:
            raise RuntimeError("did not converge")
        weight = 45 if engine == light else 40
        return SimpleNamespace(mission=SimpleNamespace(engine_model=engine.name)), object(), {"W_TO":weight}
    monkeypatch.setattr(mission_sim,"solve_engine_candidate",solve)
    selection = mission_sim.select_catalogue_engine(40000,original,[light,failed,heavy])
    assert selection.model.mission.engine_model == heavy.name
    assert selection.state["W_TO"] == 40
    assert selection.trials[1].weight is None
    assert "did not converge" in selection.trials[1].message
    assert original.engine_model == "auto"


def test_all_failed_candidates_never_produce_an_engine(monkeypatch):
    def fail(*args):
        raise ValueError("post-solve check failed")
    monkeypatch.setattr(mission_sim,"solve_engine_candidate",fail)
    selection = mission_sim.select_catalogue_engine(40000,MissionCase(),[catalogue_engine("P1000-PRO")])
    assert selection.model is None
    assert selection.state is None
    assert selection.trials[0].weight is None
