# ===========
# Description
# ===========
# Hand-calc printout of one mission pass through evaluate_one() in
# missionsweep.py, starting from the "perfect" (already converged) W_TO.
# Every equation is shown symbolically, then with numbers plugged in, then
# the result. Because the guess is the closed weight, the build-up at the end
# returns the same W_TO you started with.

import math

from missionsweep import MissionCase, Design, size_drone, sweep_designs


def step(title):
    print(f"\n{title}\n{'-' * len(title)}")


def line(name, formula, numbers, value, unit=""):
    print(f"  {name} = {formula}")
    print(f"  {' ' * len(name)} = {numbers}")
    print(f"  {' ' * len(name)} = {value:.6g} {unit}\n")


def hand_calc(m: MissionCase, d: Design, W_TO: float):
    sig, tau = d.sigma, d.tau

    step("0. Inputs")
    print(f"  W_TO (converged guess) = {W_TO:.6g} lbf")
    print(f"  sigma = {sig:.6g}   tau = {tau:.6g}   S/S_stall = {d.S_factor:.6g}   M = {m.M:g}")

    step("1. Atmosphere at cruise")
    rho = m.rho_SL * sig
    line("rho", "rho_SL * sigma", f"{m.rho_SL} * {sig:.6g}", rho, "slug/ft^3")
    h = m.h_fit * sig ** -0.5701
    line("h", "h_fit * sigma^-0.5701", f"{m.h_fit} * {sig:.6g}^-0.5701", h, "ft")
    a_trop = m.a_SL * sig ** 0.1175
    a = max(m.a_strat, a_trop)
    line("a", "max(a_strat, a_SL * sigma^0.1175)",
         f"max({m.a_strat}, {m.a_SL} * {sig:.6g}^0.1175) = max({m.a_strat}, {a_trop:.6g})", a, "ft/s")
    V = m.M * a
    line("V", "M * a", f"{m.M:g} * {a:.6g}", V, "ft/s")
    q = 0.5 * rho * V ** 2
    line("q", "0.5 * rho * V^2", f"0.5 * {rho:.6g} * {V:.6g}^2", q, "lbf/ft^2")
    lapse = m.K_ram * sig ** 0.7
    line("lapse", "K_ram * sigma^0.7", f"{m.K_ram} * {sig:.6g}^0.7", lapse)

    step("2. Supersonic drag factors (Mach held at >= 1.2)")
    M_eff = max(m.M, 1.2)
    beta = math.sqrt(M_eff ** 2 - 1)
    line("beta", "sqrt(max(M,1.2)^2 - 1)", f"sqrt({M_eff:g}^2 - 1)", beta)
    K_sup = beta / 4
    line("K_sup", "beta / 4", f"{beta:.6g} / 4", K_sup)
    K_wave = 4 / beta
    line("K_wave", "4 / beta", f"4 / {beta:.6g}", K_wave)

    step("3. Wing area from stall speed")
    S = d.S_factor * 2 * W_TO / (m.rho_SL * m.V_stall ** 2 * m.C_Lmax)
    line("S", "S_factor * 2 W_TO / (rho_SL V_stall^2 C_Lmax)",
         f"{d.S_factor:.6g} * 2 * {W_TO:.6g} / ({m.rho_SL} * {m.V_stall:g}^2 * {m.C_Lmax:g})", S, "ft^2")

    step("4. Takeoff / climb / accel fuel (energy method)")
    c_T_s = m.c_T / 3600
    line("c_T", "c_T[1/hr] / 3600", f"{m.c_T} / 3600", c_T_s, "1/s")
    E = h + V ** 2 / (2 * m.g)
    line("h_E", "h + V^2 / (2 g)   (energy height)", f"{h:.6g} + {V:.6g}^2 / (2 * {m.g})", E, "ft")
    W_f_climb = m.f_TO * W_TO + c_T_s * W_TO * E / (m.V_climb * m.eta_climb)
    line("W_f_climb", "f_TO W_TO + c_T W_TO h_E / (V_climb eta_climb)",
         f"{m.f_TO} * {W_TO:.6g} + {c_T_s:.6g} * {W_TO:.6g} * {E:.6g} / ({m.V_climb:g} * {m.eta_climb})"
         f" = {m.f_TO * W_TO:.6g} + {W_f_climb - m.f_TO * W_TO:.6g}", W_f_climb, "lbf")
    W_cr_start = W_TO - W_f_climb
    line("W_cr_start", "W_TO - W_f_climb", f"{W_TO:.6g} - {W_f_climb:.6g}", W_cr_start, "lbf")

    step("5. Cruise aerodynamics (L = W at start of cruise)")
    C_L = W_cr_start / (q * S)
    line("C_L", "W_cr_start / (q S)", f"{W_cr_start:.6g} / ({q:.6g} * {S:.6g})", C_L)
    C_D0 = m.C_f * m.Swet_S + m.C_f * m.Swet_fus / S
    line("C_D0", "C_f Swet_S + C_f Swet_fus / S",
         f"{m.C_f} * {m.Swet_S} + {m.C_f} * {m.Swet_fus:g} / {S:.6g}", C_D0)
    C_Dw = K_wave * tau ** 2 + m.CDA_fus_w / S
    line("C_Dw", "K_wave tau^2 + CDA_fus_w / S",
         f"{K_wave:.6g} * {tau:.6g}^2 + {m.CDA_fus_w} / {S:.6g}", C_Dw)
    C_Di = m.K_ind * K_sup * C_L ** 2
    line("C_Di", "K_ind K_sup C_L^2", f"{m.K_ind} * {K_sup:.6g} * {C_L:.6g}^2", C_Di)
    C_D = C_D0 + C_Dw + C_Di
    line("C_D", "C_D0 + C_Dw + C_Di", f"{C_D0:.6g} + {C_Dw:.6g} + {C_Di:.6g}", C_D)
    D = q * C_D * S
    line("D", "q C_D S", f"{q:.6g} * {C_D:.6g} * {S:.6g}", D, "lbf")
    print(f"  L/D = C_L / C_D = {C_L / C_D:.4g}")

    step("6. Engine sizing (worse of takeoff and cruise)")
    T_TO = m.TW_TO * W_TO
    T_cr = D / lapse
    line("T_SL,TO", "TW_TO W_TO", f"{m.TW_TO} * {W_TO:.6g}", T_TO, "lbf")
    line("T_SL,cr", "D / lapse", f"{D:.6g} / {lapse:.6g}", T_cr, "lbf")
    T_SL = max(T_TO, T_cr)
    line("T_SL", "max(T_SL,TO, T_SL,cr)", f"max({T_TO:.6g}, {T_cr:.6g})", T_SL, "lbf")

    step("7. Empty weight build-up")
    W_wing = m.K_wing * math.sqrt(m.N_ult * W_TO) * S ** 0.622 * m.AR ** 0.785 * tau ** -0.4
    line("W_wing", "K_wing (N_ult W_TO)^0.5 S^0.622 AR^0.785 tau^-0.4",
         f"{m.K_wing:.6g} * ({m.N_ult:g} * {W_TO:.6g})^0.5 * {S:.6g}^0.622 * {m.AR:g}^0.785 * {tau:.6g}^-0.4",
         W_wing, "lbf")
    W_eng = T_SL / m.TW_eng
    line("W_eng", "T_SL / TW_eng", f"{T_SL:.6g} / {m.TW_eng:g}", W_eng, "lbf")
    W_other = m.W_fixed + m.f_other * W_TO
    line("W_other", "W_fixed + f_other W_TO", f"{m.W_fixed:g} + {m.f_other} * {W_TO:.6g}", W_other, "lbf")
    W_empty = W_wing + W_eng + W_other
    line("W_empty", "W_wing + W_eng + W_other", f"{W_wing:.6g} + {W_eng:.6g} + {W_other:.6g}", W_empty, "lbf")
    W_ZF = m.W_pay + W_empty
    line("W_ZF", "W_pay + W_empty", f"{m.W_pay:g} + {W_empty:.6g}", W_ZF, "lbf")

    step("8. Reserve fuel and supersonic leg (Breguet endurance)")
    W_f_res = m.f_res * W_ZF
    line("W_f_res", "f_res W_ZF", f"{m.f_res} * {W_ZF:.6g}", W_f_res, "lbf")
    W_cr_end = W_ZF + W_f_res
    line("W_cr_end", "W_ZF + W_f_res", f"{W_ZF:.6g} + {W_f_res:.6g}", W_cr_end, "lbf")
    z = m.t_super * c_T_s * C_D / C_L
    line("z_bre", "t_super c_T C_D / C_L", f"{m.t_super:g} * {c_T_s:.6g} * {C_D:.6g} / {C_L:.6g}", z)
    ser = z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
    line("e^z - 1", "z + z^2/2 + z^3/6 + z^4/24", f"{z:.6g} + {z**2/2:.3g} + {z**3/6:.3g} + {z**4/24:.3g}", ser)
    W_f_cruise = W_cr_end * ser
    line("W_f_cruise", "W_cr_end (e^z - 1)", f"{W_cr_end:.6g} * {ser:.6g}", W_f_cruise, "lbf")

    step("9. Weight closure check")
    W_calc = W_ZF + W_f_climb + W_f_cruise + W_f_res
    line("W_TO,calc", "W_ZF + W_f_climb + W_f_cruise + W_f_res",
         f"{W_ZF:.6g} + {W_f_climb:.6g} + {W_f_cruise:.6g} + {W_f_res:.6g}", W_calc, "lbf")
    print(f"  Guess  W_TO      = {W_TO:.6f} lbf")
    print(f"  Result W_TO,calc = {W_calc:.6f} lbf")
    print(f"  Residual         = {W_calc - W_TO:.2e} lbf  -> closed")
    print(f"  Cruise-start check: W_cr_end + W_f_cruise = {W_cr_end + W_f_cruise:.6f}"
          f"  vs  W_cr_start = {W_cr_start:.6f}")
    print(f"  Fuel fraction  = {(W_f_climb + W_f_cruise + W_f_res) / W_TO:.4f}")
    print(f"  Empty fraction = {W_empty / W_TO:.4f}")


if __name__ == "__main__":
    mission = MissionCase()
    design, _ = sweep_designs(mission)
    W_TO = size_drone(mission, design)["W_TO"]   # the "perfect" guess
    hand_calc(mission, design, W_TO)
