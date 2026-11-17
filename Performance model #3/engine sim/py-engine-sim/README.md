# py-engine-sim

`py-engine-sim` is a small, headless Python simulator for preliminary turbojet
and nozzle studies. It is designed for parameter sweeps, optimization, and
transparent sensitivity studies. The equations follow the same reduced-order
cycle ideas used by NASA Glenn's educational EngineSim, but this is an
independent implementation rather than a line-by-line port of the Java GUI.

## What it models

- ISA atmosphere through the lower stratosphere
- Freestream stagnation conditions
- Inlet total-pressure recovery
- Single-spool dry turbojet Brayton-cycle estimate
- Compressor work and turbine work balance
- Burner temperature and pressure loss
- Convergent and convergent-divergent nozzle flow
- Gross thrust, ram drag, pressure thrust, net thrust, fuel flow, and SFC
- JSON command-line output suitable for sweeps

The model intentionally exposes assumptions that are often unavailable for a
small commercial turbojet. It is useful for comparing candidate assumptions;
it is not a JetCat performance map and it cannot replace manufacturer data,
static testing, high-speed testing, or flight safety analysis.

## Install and run

```bash
python -m pip install -e .
py-engine-sim --preset p1000 --altitude-ft 9843 --mach 1.05
```

The command prints JSON. Use `--pretty` for readable output.

```bash
py-engine-sim --preset p1000 --altitude-ft 0 --mach 0 --pretty
```

## Python API

```python
from enginesim.presets import p1000_like
from enginesim.turbojet import FlightCondition, evaluate_turbojet

engine = p1000_like()
result = evaluate_turbojet(
    engine,
    FlightCondition(altitude_ft=9843.0, mach=1.05),
    throttle=1.0,
)

print(result.net_thrust_lbf, result.fuel_flow_lbm_s)
```

## Parameter calibration

The `p1000_like()` preset separates published inputs from assumptions. The
published values are used as calibration targets, not as a claim that every
internal station variable is known. A defensible workflow is:

1. Match sea-level static thrust, airflow, and fuel flow as far as the
   available data permits.
2. Sweep inlet recovery, turbine temperature, component efficiencies, and
   nozzle throat area.
3. Run altitude/Mach sweeps and report a range, not one falsely precise value.
4. Replace assumptions with manufacturer maps or test data as they become
   available.

## Units and limitations

The public API uses imperial aerospace units: feet, inches, psf, degrees
Rankine, slug/ft^3, ft/s, lbm/s, lbf, and lbm/(lbf h). Temperatures are
reported in degrees Rankine because that is the absolute temperature scale
needed by the thermodynamic equations; `Atmosphere.temperature_f` is available
for display in degrees Fahrenheit. The solver converts to SI internally. The
cycle uses constant specific heats and simple component efficiencies. It does
not model a real compressor map,
surge, rotating stall, combustor dynamics, ECU control logic, inlet distortion,
3-D shocks, separated flow, or aeroelastic effects.

### Flow matching and calibration

The fixed nozzle sets exhaust mass flow from turbine exit total conditions.
Airflow is that flow divided by `(1 + fuel_to_air_ratio)`, so inlet air plus
fuel equals nozzle flow. Compressor pressure ratio and turbine inlet temperature
remain prescribed screening assumptions; this is not a compressor-map/spool-speed
matching calculation. The former empirical `reference_airflow_lbm_s`,
`airflow_exponent`, reference pressure/temperature and compressor-face Mach
settings are retained for API compatibility but do not determine cycle flow.

The P1000-like preset's **assumed effective throat area is 12.36037 in²**, fitted
to its 3.968 lbm/s sea-level static airflow target. It is not a published nozzle
dimension. With the current constant gas properties it predicts approximately
265.4 lbf static thrust and 1.525 lbm/(lbf h) TSFC; the thrust and fuel-flow
targets are metadata, not additional enforced calibration conditions.

The burner uses a common enthalpy reference, with `cp = gamma R/(gamma - 1)`
for each constant-property gas. Turbine shaft work balances compressor work
including mechanical efficiency. Thrust uses nozzle momentum and the actual
pressure-area term without an overall rescaling.

The nozzle uses ideal isentropic mass flow with a square-root kinetic-efficiency
correction to exit velocity. It supports convergent flow and the subsonic or
supersonic branches of a convergent-divergent nozzle. Back pressures requiring
an internal normal shock are explicitly rejected; external plume shocks and
flow separation are not modeled.

After changing cycle assumptions, run `python "engine sim/fit_lapse.py"` from
the repository root. The exploratory Mach 1 fit over 0-50 kft is
`lapse = 1.2666560 * sigma**0.9464510`, with 3.015% maximum fit error relative
to this screening cycle. The mission models use that fit and restrict the
catalogue weight fits to 4.0-43.21 lbf. Mission TSFC remains the catalogue's
static full-thrust approximation; it is not an off-design engine fuel map.
The fit error does not establish accuracy against engine measurements. Use
the mission's `--lapse` option to compare a separately assumed total thrust
ratio, and compare alternative lapse models at the same altitude and Mach.

### Shared atmosphere

`isa()` accepts geometric altitude in feet and converts it to geopotential
height using the US Standard Atmosphere 1976 radius (6,356,766 m). It solves
the linear-temperature troposphere and the isothermal lower stratosphere,
with continuous temperature, pressure and density at 11 km geopotential
height (about 36,152 geometric ft). `altitude_from_density_ratio()` provides
the analytic inverse. The mission shares these routines over 0-50 kft rather
than extrapolating an altitude power law. The engine API still supports its
previous geometric range of -5 to 20 km.

Independent reference-level tests, a dense altitude/density round trip and
tropopause continuity checks cover the shared atmosphere. A plot and CSV can
be regenerated with `python "engine sim/validate_atmosphere.py"` from the
repository root. See the
[US Standard Atmosphere 1976](https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf)
for the standard layers and altitude definitions.

## Sources

See [NOTICE.md](NOTICE.md) for NASA source links and attribution. NASA's
original pages explain that EngineSim accepts altitude, Mach, throttle, inlet,
compressor, burner, turbine, and nozzle inputs, while the nozzle tool uses
throat area, exit-area ratio, chamber total conditions, and free-stream
pressure.
