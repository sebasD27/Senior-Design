"""Conversions used by the imperial public API.

The public API uses U.S. customary aerospace units.  The solver converts to
SI internally, then converts results back at the API boundary.
"""

FT_TO_M = 0.3048
IN_TO_M = 0.0254
LBM_TO_KG = 0.45359237
LBF_TO_N = 4.4482216152605
PSF_TO_PA = LBF_TO_N / FT_TO_M**2
PSI_TO_PA = LBF_TO_N / IN_TO_M**2
R_TO_K = 5.0 / 9.0
K_TO_R = 1.0 / R_TO_K
SLUG_TO_KG = 14.59390294
BTU_TO_J = 1055.05585262
BTU_PER_LBM_TO_J_PER_KG = BTU_TO_J / LBM_TO_KG
FT_LBF_PER_SLUG_R_TO_J_PER_KG_K = (
    FT_TO_M * LBF_TO_N / (SLUG_TO_KG * R_TO_K)
)


def f_to_r(temperature_f: float) -> float:
    return temperature_f + 459.67


def r_to_f(temperature_r: float) -> float:
    return temperature_r - 459.67
