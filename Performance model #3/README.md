# Performance model #3

Weight-closure and performance sizing for the Mach 1 turbojet drone. Run everything from
`weigth closure/` with the `esp312` environment (it has `lcsolver`).

Changes since model #2: input file (`inputs.py`), dash altitude as a GP variable, descent and landing
segments, rail launch / landing-weight wing sizing, CFD-fitted transonic drag rise and lift-dependent
drag, quarter-chord wing weight, fuselage volume check (engine, duct, fuel, systems), conservative thrust
lapse with installed-thrust losses, engine clearance, dynamic-pressure limit, the ESP geometry tools in
`geometry/`, and a lowest-dash-altitude study.

## Current design (`python inputs.py`)

39.9 lb, JetCat P400-PRO-LN, Mach 1 dash at 30,000 ft (the lowest altitude that closes with these
inputs), 5 % thrust margin. Wing 1.55 ft^2, AR 2, 63 deg LE sweep (delta limit), t/c 0.03; fuselage
7.4 ft x 6.8 in. Mach 1 drag 37 lbf, L/D 0.97. `k_pod` and `k_fin` are still placeholders that carry about
a quarter of the drag; the testDrone CAD values are much lower.

## Layout

| Folder | What it holds |
|---|---|
| `weigth closure/` | Core sizing code (below) |
| `weigth closure/studies/` | Trade studies and plots built on the core code |
| `weigth closure/tests/` | Tests (`python -m pytest tests`; optimizer tests need `lcsolver`) |
| `geometry/` | ESP model `testDrone.csm`, `testDrone_optimal.csm` and `make_optimal_csm.py` (resizes the CSM to an optimizer result) |
| `engine sim/` | Turbojet cycle model (`py-engine-sim`), thrust-lapse map and fits |
| `propulsion/` | Engine catalogue (`engine_catalogue.csv`) and its weight/thrust/TSFC/diameter fits |
| `archive/` | Superseded or broken files and plots from earlier model versions; not used by any code |

### Core files (`weigth closure/`)

| File | Role |
|---|---|
| `inputs.py` | Input file: every knob in one place. `python inputs.py` solves and prints the report |
| `mission_sim.py` | GP optimizer (lcsolver); `run_case()` solves and reports for `inputs.py` and the command line |
| `missionsweep.py` | `MissionCase` (all mission and model constants) and the numeric fixed-point closure |
| `aero.py` | Component drag build-up: friction, wave (with CFD-fitted transonic rise), base, pod, interference, lift |
| `mission_path.py` | Takeoff, climb and acceleration segments; installed thrust along the path |
| `gp_atmosphere.py` | Exact ISA troposphere in GP form (altitude as a variable); idle-descent fuel factor |
| `mission_atmosphere.py` | Exact ISA lookups shared by everything |
| `engine_choice.py` | Catalogue engines, lapse map, command-line hardware options |
| `csm_geometry.py` | Converts testDrone CSM parameters into optimizer inputs |
| `breakdown.py` | Weight and drag breakdown printout |
| `altitude_search.py` | Old fixed-altitude sweep, still available with `--altitude-search` |
| `aero_surrogate.py` | Evaluates the SU2 Euler Gaussian-process surrogates (`gp_M*_a*.json`) |

### Studies (`weigth closure/studies/`)

`lowest_altitude.py` (lowest dash altitude that closes; pin an engine in `inputs.py`), `sweep_lapse.py`,
`sweep_tradeoff.py`, `sensitivity.py`, `drag_vs_length.py`, `drag_vs_speed.py`. Plots go to
`weigth closure/plots/`.

## Running

```bash
cd "weigth closure"
python inputs.py                            # the input file: weight, drag and planform report + plot
python studies/lowest_altitude.py           # lowest closing dash altitude for inputs.py
python mission_sim.py --help                # command-line version and its options
```

With `ALTITUDE = None` the GP chooses the dash altitude (troposphere) and the exact fixed-altitude model is
re-solved there; the printed report is the exact one.

## Mission

Rail launch (or takeoff on legs), sea-level acceleration, climb, level acceleration to Mach 1 at the dash
altitude with a thrust margin, 5 s level dash, idle descent, approach and landing on legs, 6 % reserve.
No return-to-base cruise.

## Inputs still to set

All in `weigth closure/inputs.py`.

| Input | Now | Note |
|---|---|---|
| `k_pod`, `k_fin` | 0.3, 0.25 | Placeholders; testDrone gives about 0.055 and 0.06 (about 25 % less drag) |
| `f_idle` | 0.20 | Idle / full-thrust fuel flow at sea level; take from the engine datasheet |
| `eta_inst`, `d_clear`, `f_vol_use`, `eng_LD`, `V_sys` | placeholders | Installation and layout; set from the CAD and engine data |
| `q_max` | 800 psf | Structural design dynamic pressure (better: weights and flutter as functions of q) |
| `k_fric` | 0.613 | Friction calibrated to a hand estimate; the Euler CFD cannot check friction |
| `W_fixed`, `f_other` | 3 lb, 20 % | Replace with a real avionics / systems weight list |
| Static margin | not modelled | Needs component positions; check it in the ESP layout |
