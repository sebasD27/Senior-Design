"""Shared flight segments for finite-time climb/acceleration fuel accounting.

Start/takeoff fuel covers liftoff at 1.2 V_stall. Then accelerate at sea level,
climb at V_climb, and accelerate at cruise altitude. Each interval checks both
endpoints and its midpoint using the same atmosphere, lapse and drag routines.
"""
from dataclasses import dataclass
import math

import numpy as np

from aero import drag_terms
from engine_choice import lapse_map, mission_lapse
from mission_atmosphere import atmosphere_at_altitude, density_ratio


def thrust_lapse(m, altitude, mach):
    if m.lapse_override is not None:
        # A user override is explicitly a constant scenario over the path.
        return mission_lapse(density_ratio(altitude), m.K_ram, m.lapse_exp, m.lapse_override)
    fitted_mach_one = mission_lapse(density_ratio(altitude), m.K_ram, m.lapse_exp)
    return fitted_mach_one * lapse_map(altitude, mach) / lapse_map(altitude, 1.0)


@dataclass(frozen=True)
class FlightPoint:
    altitude: float
    speed: float
    density: float
    mach: float
    lapse: float

    @property
    def q(self):
        return 0.5 * self.density * self.speed**2


def flight_point(m, altitude, speed):
    state = atmosphere_at_altitude(altitude)
    mach = speed / state.speed_of_sound_ft_s
    return FlightPoint(altitude, speed, state.density_slug_ft3, mach, thrust_lapse(m, altitude, mach))


@dataclass(frozen=True)
class Segment:
    phase: str
    points: tuple
    # time >= W * time_coefficient / (T - D), with W and T in lbf.
    time_coefficient: float
    minimum_time: float


def flight_segments(m, altitude, intervals=8):
    if intervals < 1:
        raise ValueError("flight path needs at least one interval per phase")
    final_speed = m.M * atmosphere_at_altitude(altitude).speed_of_sound_ft_s
    climb_speed = min(m.V_climb, final_speed)
    liftoff_speed = 1.2 * m.V_stall
    if climb_speed < liftoff_speed:
        raise ValueError("mission speed must exceed the modeled liftoff speed")
    segments = []

    def accelerate(phase, height, start, end):
        if end <= start:
            return
        speeds = np.linspace(start, end, intervals + 1)
        for low, high in zip(speeds[:-1], speeds[1:]):
            points = tuple(flight_point(m, height, float(v)) for v in (low, (low + high) / 2, high))
            segments.append(Segment(phase, points, float(high - low) / m.g, 0.0))

    accelerate("low-altitude acceleration", 0.0, liftoff_speed, climb_speed)
    if altitude > 0:
        heights = np.linspace(0, altitude, intervals + 1)
        for low, high in zip(heights[:-1], heights[1:]):
            points = tuple(flight_point(m, float(h), climb_speed) for h in (low, (low + high) / 2, high))
            dh_over_v = float(high - low) / climb_speed
            segments.append(Segment("climb", points, dh_over_v, dh_over_v))
    accelerate("high-altitude acceleration", altitude, climb_speed, final_speed)
    return segments


def point_drag(m, point, *, weight, S, tau, c_sw, L, d):
    terms, _ = drag_terms(m, S=S, tau=tau, c_sw=c_sw, L=L, d=d,
                          rho=point.density, V=point.speed, M=point.mach,
                          C_L=weight / (point.q * S))
    return point.q * S * sum(terms.values())


def segment_loads(m, segments, **geometry):
    """Numerical loads for sizing and post-solve verification."""
    result = []
    for segment in segments:
        drag = max(float(point_drag(m, p, **geometry)) for p in segment.points)
        result.append((segment, drag, min(p.lapse for p in segment.points), max(p.lapse for p in segment.points)))
    return result


def path_fuel(m, loads, weight, thrust_sl, tsfc):
    """Conservative segment estimates: minimum thrust, maximum sampled drag.

    Use takeoff weight throughout and the maximum segment thrust for fuel.
    Static full-thrust TSFC is still a screening assumption.
    """
    fuel = 0.0
    elapsed = 0.0
    for segment, drag, low_lapse, high_lapse in loads:
        available = thrust_sl * low_lapse
        if available <= drag or available < (1 + m.accel_margin) * drag * (1 - 1e-7):
            return math.inf, math.inf
        time = max(segment.minimum_time, weight * segment.time_coefficient / (available - drag))
        elapsed += time
        fuel += tsfc / 3600 * thrust_sl * high_lapse * time
    return fuel, elapsed
