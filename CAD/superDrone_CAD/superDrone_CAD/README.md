# superDrone — SolidWorks CAD assembly

Generated from `superDrone_inlet_tefix.csm`. The outer mold line (fuselage ogive + cap, cylinder, boattail,
NACA delta wing, truncated-TE fin, strakes, ventral cowl and S-duct) uses the .csm parameters and formulas
one-to-one. Everything inside it is a new conceptual layout.

## Files

| File | What it is |
|---|---|
| `superDrone_gear_down.step` | Full assembly, gear extended, gear doors open |
| `superDrone_gear_up.step` | Full assembly, gear retracted, doors closed |
| `build_superdrone.py`, `sd_params.py`, `sd_geom.py`, `sd_airframe.py` | CadQuery generator (change a parameter, re-run) |
| `check_interference.py` | Validity + pairwise interference check |
| `preview_*.png` | Rendered previews |

## Opening in SolidWorks

File → Open → STEP (*.step). In import options keep "Import multiple bodies as parts" so the product
structure comes in as an assembly with sub-assemblies (Airframe, Structure, Propulsion, Fuel system,
Landing gear, Avionics). All parts arrive as dumb solids with names and colours, no mates.

* **Units:** millimetres. The .csm is unitless; it was read as **metres** (11.84 m nose-tip to nozzle, 2.6 m span).
  If you meant a different unit, change `scale=1000.0` in `to_assembly()` and re-run.
* **Axes:** the .csm build frame (su2:axes = 0): **+X aft** (nose at −X), **+Y up**, **+Z starboard**.
  This matches SolidWorks' Y-up default, so the Front view looks at the starboard side.
* **Ground plane:** all three tyres touch Y = −750 mm in the gear-down file.

### Making the gear move
Add a concentric mate on each trunnion pin and a hinge about these axes (gear up = −90° about +Z,
i.e. the wheels swing forward):

| Leg | Pivot (mm) | Axis | Retract |
|---|---|---|---|
| Nose | (−4650, −60, 0) | Z | −90° (forward) |
| Main stbd | (1620, −20, 240) | Z | −90° |
| Main port | (1620, −20, −240) | Z | −90° |

Main doors hinge along X at (Y 0, Z ±300); nose doors along X at roughly (Y −238, Z ±55).
The actuators are modelled separately per configuration (barrel + rod), so they are correct in both files
but will not follow a mate in SolidWorks unless you mate rod-in-barrel.

## Internal layout (stations in m, build frame)

| Station x | Item |
|---|---|
| −8.14 … −7.81 | Pitot-static probe |
| −7.30 / −5.48 | Avionics bay bulkheads B1/B2: ADC, FCC, INS/GNSS, datalink radio, PDU, avionics battery on a tray |
| −5.40 … −4.50 | Nose-gear bay, 2 doors; NLG pivot −4.65, retracts forward |
| −4.38 / −2.05 | Tank-1 bulkheads B3/B4; **tank 1** conformal, 452 L |
| −1.95 … −0.12 | **Tank 2**, saddle over the S-duct, 385 L |
| −1.60 | Inlet lip |
| 0.05 … 0.85 | Equipment bay: fuel pump, filter, shut-off valve, engine ECU, main battery, gear/servo controller |
| 0.68 / 1.48 | Wing front / rear spar frames (F7/F8), root fittings through the skin |
| 0.78 … 1.82 | Main-gear side bays + doors; MLG pivot 1.62, retracts forward |
| 1.74 / 2.11 | Fin spar frames (F9/F10); F9 also carries the MLG trunnion fitting |
| 2.40 | Engine face (duct end) |
| 2.42 … 3.33 | **Turbojet**: intake/compressor casing, starter bullet, centrifugal impeller, diffuser, annular combustor, 8 injectors + manifold, NGV, turbine wheel, exhaust cone; mounts at F11 (2.60) and F12 (3.20) |
| 3.33 … 3.70 | Tailpipe / convergent nozzle (exit radius 158 mm from `inlet:exitRatio`) |

Total internal fuel ≈ **837 L (≈ 670 kg Jet-A)**. To get mass and CG in SolidWorks, assign Jet-A
density (800 kg/m³) to the two tank bodies, which are modelled as solid fuel volumes.

Structure: 4 bulkheads, 9 ring frames, 4 longerons, 2 spars + 5 ribs per wing, 2 spars + 4 ribs in the fin.
Skins: fuselage 3 mm, cowl 4 mm, duct wall 10 mm, wing/fin 2 mm.

## Checks run
* All 153 solids valid (BRepCheck) in both files.
* Pairwise interference check in both configurations: **0 interferences** gear-down and gear-up (`check_down.log`, `check_up.log`).

## Design flags worth looking at
1. **Main-gear track is only 0.48 m** on a 2.6 m span. The 5 % wing (90 mm deep at the root) cannot
   house a wheel, so the mains stow in fuselage side bays beside the duct. Lateral tip-over margin is low;
   a wider track would need wing-root pods/fairings.
2. **The S-duct crosses the lower longeron line between F6 and F8**, so the lower longerons are split there.
   Bending in that bay has to go through the duct wall and frames; size it accordingly.
3. The S-duct also passes through the wing-root carry-through zone, so the wings attach through
   spar-frame fittings rather than a through-spar.
4. Engine envelope (~360 mm dia × 0.9 m) is sized to the .csm engine-face radius, not a specific engine.
   Swap in a vendor model when you choose one.
5. Not modelled: the flend fillets (plain intersection at the wing/fin roots), control surfaces
   (`controls = 0` in the .csm), wiring, plumbing beyond the main feed path.
