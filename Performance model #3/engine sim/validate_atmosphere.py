"""Print and plot the shared 0-50 kft atmosphere against the former mission fit.

Standard layers: US Standard Atmosphere 1976, geometric altitude input.
Run: python "engine sim/validate_atmosphere.py"
"""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "py-engine-sim" / "src"))
from enginesim.atmosphere import isa


def main():
    altitudes = np.linspace(0, 50000, 1001)
    states = [isa(h) for h in altitudes]
    rho0 = isa(0).density_slug_ft3
    sigma = np.array([s.density_slug_ft3 / rho0 for s in states])
    speed = np.array([s.speed_of_sound_ft_s for s in states])
    old_height = 7720.7 * sigma**-1.4916
    old_speed = np.maximum(968.1, 1116.4 * sigma**0.1175)
    output = HERE / "plots"
    output.mkdir(exist_ok=True)
    with (output / "atmosphere_validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["geometric_altitude_ft", "temperature_R", "pressure_psf", "density_slug_ft3", "sound_speed_ft_s", "sigma", "old_fit_implied_altitude_ft"])
        for state, ratio, old in zip(states, sigma, old_height):
            writer.writerow([state.altitude_ft, state.temperature_r, state.pressure_psf, state.density_slug_ft3, state.speed_of_sound_ft_s, ratio, old])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(sigma, altitudes / 1000, label="Layered standard atmosphere")
    axes[0].plot(sigma, old_height / 1000, "--", label="Former mission fit / extrapolation")
    axes[0].set(xlabel="Density / sea-level density", ylabel="Geometric altitude (kft)", ylim=(0, 50))
    axes[0].legend(fontsize=8)
    axes[1].plot(altitudes / 1000, speed, label="Layered standard atmosphere")
    axes[1].plot(altitudes / 1000, old_speed, "--", label="Former sound-speed fit")
    axes[1].set(xlabel="Geometric altitude (kft)", ylabel="Sound speed (ft/s)")
    axes[1].legend(fontsize=8)
    axes[2].plot(altitudes / 1000, old_height - altitudes)
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set(xlabel="Geometric altitude (kft)", ylabel="Former altitude-fit error (ft)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "atmosphere_validation.png", dpi=150)
    plt.close(fig)
    print("Geometric ft   T (R)   p (psf)   rho (slug/ft^3)   a (ft/s)   sigma")
    for h in range(0, 50001, 10000):
        s = isa(h)
        print(f"{h:12,d} {s.temperature_r:7.2f} {s.pressure_psf:9.2f} {s.density_slug_ft3:17.8f} {s.speed_of_sound_ft_s:10.2f} {s.density_slug_ft3/rho0:7.4f}")
    print(f"Former altitude fit reports {old_height[0]:.1f} ft at sea-level density and {old_height[-1]:.1f} ft at 50-kft density.")
    print(f"Saved {output / 'atmosphere_validation.png'} and CSV")


if __name__ == "__main__":
    main()
