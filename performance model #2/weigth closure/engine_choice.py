"""Catalogue engine inputs shared by the mission optimizers.

Catalogue weight may include integrated accessories. Additional installation
weight must cover only hardware not already in that published weight or in
the mission's other-weight allowance.
"""
import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Engine:
    name: str
    weight: float          # lb (numerically lbf at standard gravity)
    thrust: float          # sea-level static lbf
    tsfc: float            # lb/(lbf hr), static full thrust
    diameter: float        # ft, engine envelope only


@lru_cache(maxsize=None)
def catalogue_engine(name: str) -> Engine:
    path = Path(__file__).resolve().parents[1] / "propulsion" / "engine_catalogue.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if name.casefold() in (row["engine"].casefold(), f"{row['manufacturer']} {row['engine']}".casefold()):
            try:
                engine = Engine(f"{row['manufacturer']} {row['engine']}", float(row["weight_lb"]),
                                float(row["thrust_max_lbf"]), float(row["tsfc_lb_per_lbf_h"]),
                                float(row["diameter_mm"]) / 304.8)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{name}: catalogue needs weight, thrust, TSFC and diameter") from error
            if not all(math.isfinite(v) and v > 0 for v in (engine.weight, engine.thrust, engine.tsfc, engine.diameter)):
                raise ValueError(f"{name}: catalogue values must be finite and positive")
            return engine
    raise ValueError(f"Unknown engine {name!r}; use an engine name from {path.name}")


def catalogue_candidates():
    """Return complete catalogue engines and explicit reasons for skipped rows.

    Weight, thrust, TSFC and diameter stay paired to one engine. Missing
    diameters are not filled from a regression during catalogue selection.
    """
    path = Path(__file__).resolve().parents[1] / "propulsion" / "engine_catalogue.csv"
    with path.open(newline="") as stream:
        names = [f"{r['manufacturer']} {r['engine']}" for r in csv.DictReader(stream)]
    candidates, skipped = [], []
    for name in names:
        try:
            candidates.append(catalogue_engine(name))
        except ValueError as error:
            skipped.append((name, str(error)))
    return candidates, skipped


@lru_cache(maxsize=None)
def _lapse_grid() -> np.ndarray:
    """T / T_SLS from engine sim/lapse_contour.py: rows 0-50,000 ft (101), columns Mach 0-1.5 (61)."""
    path = Path(__file__).resolve().parents[1] / "engine sim" / "plots" / "lapse_grid.csv"
    grid = np.loadtxt(path, delimiter=",")
    if grid.shape != (101, 61) or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("lapse grid must cover 0-50 kft and Mach 0-1.5 with positive finite ratios; regenerate lapse_contour.py")
    return grid


def lapse_map(altitude_ft: float, mach: float) -> float:
    """Screening thrust lapse, bilinear over 0-50 kft and Mach 0-1.5."""
    if not math.isfinite(altitude_ft) or not 0 <= altitude_ft <= 50000 or not math.isfinite(mach) or not 0 <= mach <= 1.5:
        raise ValueError("lapse map supports 0-50,000 ft and Mach 0-1.5; extrapolation is not supported")
    grid = _lapse_grid()
    i = min(max(altitude_ft, 0.0), 50000.0) / 500.0
    j = min(max(mach, 0.0), 1.5) / 0.025
    i0, j0 = min(int(i), grid.shape[0] - 2), min(int(j), grid.shape[1] - 2)
    fi, fj = i - i0, j - j0
    return float((1 - fi) * ((1 - fj) * grid[i0, j0] + fj * grid[i0, j0 + 1])
                 + fi * ((1 - fj) * grid[i0 + 1, j0] + fj * grid[i0 + 1, j0 + 1]))


def nonnegative_weight(value: str) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise argparse.ArgumentTypeError("weight must be finite and nonnegative (lb)")
    return weight


def positive_lapse(value: str) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0:
        raise argparse.ArgumentTypeError("lapse must be a finite positive net-thrust / SLS-thrust ratio")
    return ratio


def mission_lapse(sigma: float, coefficient: float, exponent: float, override: float | None = None) -> float:
    """Total thrust ratio, distinct from the coefficient of a density fit.

    An override is a scenario assumption at the mission condition. It is not
    an inlet pressure recovery or a new altitude-dependent performance map.
    """
    ratio = override if override is not None else coefficient * sigma**exponent
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("mission thrust lapse must be finite and positive")
    return ratio


def add_hardware_arguments(parser):
    parser.add_argument("--lapse", type=positive_lapse, default=None,
                        help="override total cruise net thrust / SLS thrust (e.g. 0.179); scenario assumption, not K_ram or inlet recovery")
    parser.add_argument("--engine", help='hold a catalogue engine fixed, e.g. "P1000-PRO"')
    parser.add_argument("--installation-weight", type=nonnegative_weight, default=0.0,
                        help="additional propulsion installation hardware, lb; avoid counting existing allowances twice")
    parser.add_argument("--payload-weight", type=nonnegative_weight, default=0.0, help="payload, lb")
