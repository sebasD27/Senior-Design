# Shared setup for the studies: baseline from inputs.py with a relaxed weight cap, exact fixed-altitude
# solves with a pinned catalogue engine, a parallel map, and one plot style.
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # core modules live one folder up

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inputs
from mission_atmosphere import atmosphere_at_altitude
from mission_sim import build_mission, solve_mission

W_CAP = 150.0          # relaxed takeoff-weight cap for exploring the design space
W_LIMIT = 55.0         # the real requirement (Part 107), drawn as a reference line
# Catalogue engines spanning the useful range (the continuous engine fit misses designs these close)
ENGINES = ["P220-PRO", "P400-PRO-LN", "P550-PRO-S", "P900-PRO-S-O-R", "P1000-PRO"]
PLOT_DIR = HERE.parent / "plots" / "studies"

# Reference categorical palette (dataviz skill, light mode), fixed order; text and grid in neutral inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def baseline(**changes):
    """MissionCase from inputs.py, weight cap relaxed, with any field changes applied."""
    return replace(inputs.mission_case(), W_max=W_CAP, **changes)


def solve(job):
    """job = (label, altitude_ft, MissionCase, engine[, initial state]). engine None keeps the continuous
    engine fit. Returns (label, altitude, engine, state or None, note)."""
    label, h, m, engine, *rest = job
    try:
        _, _, r = solve_mission(build_mission(h, replace(m, engine_model=engine)), rest[0] if rest else None)
        return label, h, engine, r, ""
    except (RuntimeError, ValueError) as error:
        return label, h, engine, None, str(error).split("\n")[0][:80]


def run_all(jobs, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(solve, jobs))


def scaled_engine(name="P400-PRO-LN"):
    """MissionCase engine-fit fields for a continuously scalable copy of a catalogue engine: thrust in
    proportion to weight (T_int = 0), its TSFC, and diameter on the catalogue exponent through its size."""
    from engine_choice import catalogue_engine
    e = catalogue_engine(name)
    d_exp = inputs.mission_case().d_exp
    return dict(T_slope=e.thrust / e.weight, T_int=0.0, tsfc_int=e.tsfc, tsfc_slope=0.0,
                d_coef=e.diameter / e.weight ** d_exp, W_eng_min=1.0)


def dash_q(h, mach):
    a = atmosphere_at_altitude(h)
    return 0.5 * a.density_slug_ft3 * (mach * a.speed_of_sound_ft_s) ** 2


def altitude_at_q(q, mach, lo=0.0, hi=50000.0):
    """Dash altitude where the dynamic pressure equals q (bisection)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dash_q(mid, mach) > q else (lo, mid)
    return 0.5 * (lo + hi)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
