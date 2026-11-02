"""Cheap validation checks: (1) inlet-recovery sensitivity of thrust lapse,
(2) independent skin-friction estimate vs the model's C_f = 0.003."""
import math, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "py-engine-sim" / "src"))
from enginesim.presets import p1000_like
from enginesim.turbojet import FlightCondition, evaluate_turbojet
from enginesim.atmosphere import isa

eng = p1000_like()
T0 = evaluate_turbojet(eng, FlightCondition(0, 0)).net_thrust_lbf

def lapse(mach, alt, rec):
    e = replace(eng, inlet_pressure_recovery=rec)
    return evaluate_turbojet(e, FlightCondition(alt, mach)).net_thrust_lbf / T0

def mil(m):  # MIL-E-5007D inlet recovery
    return 1.0 if m <= 1 else 1 - 0.075 * (m - 1) ** 1.35

print("1) Inlet pressure recovery sensitivity, lapse at 20,000 ft (T_SLS = %.1f lbf)" % T0)
print("   recovery      M0.8   M1.0   M1.2")
for rec in (0.96, 0.92, 0.88, 0.85, 0.80):
    print(f"   {rec:<12.2f}" + "".join(f"{lapse(m, 20000, rec):7.3f}" for m in (0.8, 1.0, 1.2)))
print("   MIL-E-5007D " + "".join(f"{lapse(m, 20000, min(mil(m), 1.0)):7.3f}" for m in (0.8, 1.0, 1.2)))
sigma20 = isa(20000).density_slug_ft3 / isa(0).density_slug_ft3
print(f"   (0-50 kft screening fit K_ram*sigma^0.9464510 = {1.2666560 * sigma20**0.9464510:.3f} at 20,000 ft)")

print("\n2) Turbulent flat-plate skin friction, 20,000 ft, M 1.0 (Raymer eq., Mach-corrected)")
atm20 = isa(20000)
T, rho, a = atm20.temperature_r, atm20.density_slug_ft3, atm20.speed_of_sound_ft_s
mu = 2.27e-8 * T**1.5 / (T + 198.72) * 1.0   # Sutherland, slug/(ft s)
for name, L in (("fuselage 9.96 ft", 9.96), ("wing MAC 1.38 ft", 1.382)):
    Re = rho * a * L / mu
    cf = 0.455 / (math.log10(Re) ** 2.58 * (1 + 0.144 * 1.0**2) ** 0.65)
    print(f"   {name:<18} Re = {Re:.3g}   C_f = {cf:.5f}")
print("   Model uses C_f = 0.003 on all wetted area (no separate form factor).")
