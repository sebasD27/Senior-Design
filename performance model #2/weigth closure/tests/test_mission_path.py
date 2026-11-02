"""Independent force/time/fuel checks for the shared segment accounting."""
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission_path import Segment, path_fuel, flight_segments
from missionsweep import MissionCase
from engine_choice import lapse_map


def test_segment_time_and_fuel_follow_excess_thrust():
    segment = Segment("acceleration", (), time_coefficient=2.0, minimum_time=3.0)
    # W=100 lb, Tmin=50 lbf, D=30 lbf: time = 100*2/(50-30)=10 s.
    # Fuel conservatively uses Tmax=60 lbf: 1.2/3600*60*10=0.2 lb.
    fuel, time = path_fuel(SimpleNamespace(accel_margin=.1), [(segment,30,.5,.6)],100,100,1.2)
    assert time == pytest.approx(10)
    assert fuel == pytest.approx(.2)


def test_vertical_speed_cap_limits_climb_time():
    segment = Segment("climb", (), time_coefficient=2.0, minimum_time=2.0)
    fuel, time = path_fuel(SimpleNamespace(accel_margin=.1), [(segment,10,1,1)],10,100,1.2)
    assert time == 2.0
    assert fuel == pytest.approx(1.2/3600*100*2)


def test_no_excess_thrust_has_no_finite_path_solution():
    segment = Segment("acceleration", (), 1.0, 0.0)
    fuel, time = path_fuel(SimpleNamespace(accel_margin=.1), [(segment,50,.5,.5)],100,100,1.2)
    assert math.isinf(fuel) and math.isinf(time)


def test_zero_altitude_route_has_no_climb_but_keeps_acceleration():
    m = MissionCase(engine_model="P1000-PRO", lapse_override=.179)
    route = flight_segments(m,0)
    assert route
    assert not any(s.phase == "climb" for s in route)
    assert all(p.lapse == .179 for s in route for p in s.points)
    assert route[0].points[0].speed == pytest.approx(1.2*m.V_stall)


@pytest.mark.parametrize("height,mach", [(50001,1),(40000,1.51),(-1,0),(0,float('nan'))])
def test_thrust_map_refuses_extrapolation(height,mach):
    with pytest.raises(ValueError, match="extrapolation"):
        lapse_map(height,mach)
