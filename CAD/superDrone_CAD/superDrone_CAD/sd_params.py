"""superDrone parameters - transcribed 1:1 from superDrone_inlet_tefix.csm (build frame).

Build frame (same as the .csm with su2:axes = 0):
    x  aft (nose at -x, freestream toward +x)
    y  up
    z  starboard (right wing on +z)
Lengths are in metres here; the STEP export is scaled to millimetres.
"""
import math

def tand(d): return math.tan(math.radians(d))
def sind(d): return math.sin(math.radians(d))
def cosd(d): return math.cos(math.radians(d))
def atand(v): return math.degrees(math.atan(v))
def atan2d(a, b): return math.degrees(math.atan2(a, b))

flendSize = 0.10

# ---------------- wing ----------------
wing_span, wing_croot, wing_taper, wing_tesweep = 1.3, 1.87, 0.383, 30.0
wing_dihed, wing_xroot, wing_yloc = 0.0, 0.0, 0.1
wing_thickr = wing_thickt = 0.05
wing_ctip = wing_croot * wing_taper
wing_xtip = wing_xroot + wing_croot + wing_span * tand(wing_tesweep) - wing_ctip
wing_ytip = wing_yloc + wing_span * tand(wing_dihed)
wing_sweep = atand((wing_xtip - wing_xroot) / wing_span)
lam = wing_ctip / wing_croot
wing_refAreaHalf = wing_span * (wing_croot + wing_ctip) / 2
wing_refArea = 2 * wing_refAreaHalf
wing_MAC = (2 / 3) * wing_croot * (1 + lam + lam ** 2) / (1 + lam)
wing_yMAC = (wing_span / 3) * (1 + 2 * lam) / (1 + lam)
wing_xMAC = wing_xroot + (wing_xtip - wing_xroot) * wing_yMAC / wing_span
wing_xRef = wing_xMAC + 0.25 * wing_MAC

def wing_xle(z): return wing_xroot + (wing_xtip - wing_xroot) * z / wing_span
def wing_c(z): return wing_croot + (wing_ctip - wing_croot) * z / wing_span

# ---------------- fuselage ----------------
fus_radius, fus_lnose, fus_rnose = 0.3, 7.0, 0.10
fus_ltail, fus_rtail, fus_mfwd, fus_maft, fus_maftMargin = 1.0, 0.24, 2.0, 0.8, 0.15
fus_xcyl1 = wing_xroot - fus_mfwd
fus_maftEff = max(fus_maft, (fus_radius + flendSize) * tand(wing_tesweep) + fus_maftMargin)
fus_xcyl2 = wing_xroot + wing_croot + fus_maftEff
fus_xnose = fus_xcyl1 - fus_lnose
fus_xtail = fus_xcyl2 + fus_ltail

# ---------------- vertical fin ----------------
vt_height, vt_croot, vt_taper, vt_sweep, vt_thick = 1.1, 1.0, 0.4, 45.0, 0.08
vt_xinset, vt_tekeep = 0.4, 0.92
vt_ctip = vt_croot * vt_taper
vt_xroot = fus_xcyl2 - vt_xinset - vt_croot
vt_xtip = vt_xroot + vt_height * tand(vt_sweep)

def fin_xle(h): return vt_xroot + (vt_xtip - vt_xroot) * h / vt_height
def fin_c(h): return vt_croot + (vt_ctip - vt_croot) * h / vt_height

# ---------------- strakes ----------------
st_lfwd, st_eta, st_caft, st_ctip, st_over, st_thick = 1.6, 0.55, 0.6, 0.35, 0.05, 0.03
st_zs = st_eta * wing_span
st_xt = wing_xroot + st_zs * (wing_xtip - wing_xroot) / wing_span + st_over
st_yt = wing_yloc + st_zs * tand(wing_dihed)
st_cr = st_lfwd + st_caft

# ---------------- inlet ----------------
in_exitRatio, in_xoff, in_w, in_h, in_nsup = 0.8, 0.4, 0.24, 0.10, 2.5
in_blend, in_lipt, in_gap, in_lstraight, in_xface, in_efRatio = 0.96, 0.02, 0.005, 0.5, 2.4, 1.3
in_xlip = fus_xcyl1 + in_xoff
in_xs = in_xlip + in_lstraight
in_lext = min(0.3, 0.5 * in_xoff)
in_ref = math.sqrt(in_efRatio * in_w * in_h)
in_rex = min(in_ref * math.sqrt(in_exitRatio), fus_rtail - 0.03)
in_N = 8
in_Rb = in_Rf = fus_radius
in_yc0 = -in_Rb - in_gap - in_h

# ---------------- ogive nose ----------------
og_len = fus_xcyl1 - fus_xnose
og_rho = (fus_radius ** 2 + og_len ** 2) / (2 * fus_radius)
og_cy = fus_radius - og_rho
og_xs = fus_xcyl1 - math.sqrt((og_rho - fus_rnose) ** 2 - og_cy ** 2)
og_thT = atan2d(-og_cy, og_xs - fus_xcyl1)
og_xT = fus_xcyl1 + og_rho * cosd(og_thT)
og_yT = og_cy + og_rho * sind(og_thT)
og_xtip = og_xs - fus_rnose


def r_oml(x):
    """Outer fuselage radius (no inlet cowl) at station x."""
    if x < og_xtip or x > fus_xtail:
        return 0.0
    if x < og_xT:
        return math.sqrt(max(0.0, fus_rnose ** 2 - (x - og_xs) ** 2))
    if x < fus_xcyl1:
        return og_cy + math.sqrt(max(0.0, og_rho ** 2 - (x - fus_xcyl1) ** 2))
    if x <= fus_xcyl2:
        return fus_radius
    return fus_radius + (fus_rtail - fus_radius) * (x - fus_xcyl2) / fus_ltail


def inlet_station(k):
    """Return (x, yc, n, s) for S-bend station k = 1..N."""
    t = k / in_N
    s = t * t * (3 - 2 * t)
    return (in_xs + t * (in_xface - in_xs), in_yc0 * (1 - s), in_nsup + s * (2 - in_nsup), s)


def duct_sections(x_start, grow=0.0):
    """Flow-passage sections (x, yc, a, b, n), enlarged by `grow` (wall/clearance)."""
    secs = [(x_start, in_yc0, in_w + grow, in_h + grow, in_nsup),
            (in_xs, in_yc0, in_w + grow, in_h + grow, in_nsup)]
    for k in range(1, in_N + 1):
        x, yc, n, s = inlet_station(k)
        secs.append((x, yc, in_w + s * (in_ref - in_w) + grow, in_h + s * (in_ref - in_h) + grow, n))
    return secs


def cowl_sections(x_start=None, shrink=0.0):
    a0, b0 = in_w + in_lipt - shrink, in_h + in_lipt - shrink
    secs = [(in_xlip if x_start is None else x_start, in_yc0, a0, b0, in_nsup),
            (in_xs, in_yc0, a0, b0, in_nsup)]
    for k in range(1, in_N + 1):
        x, yc, n, s = inlet_station(k)
        ao = in_w + in_lipt + s * (in_blend * in_Rf - in_w - in_lipt) - shrink
        bo = in_h + in_lipt + s * (in_blend * in_Rf - in_h - in_lipt) - shrink
        secs.append((x, yc, ao, bo, n))
    return secs


def duct_at(x, grow=0.0):
    """Interpolated duct section (yc, a, b, n) at station x (for clearance checks)."""
    secs = duct_sections(in_xlip - in_lext, grow)
    for (x0, y0, a0, b0, n0), (x1, y1, a1, b1, n1) in zip(secs[:-1], secs[1:]):
        if x0 <= x <= x1:
            f = (x - x0) / (x1 - x0) if x1 > x0 else 0
            return (y0 + f * (y1 - y0), a0 + f * (a1 - a0), b0 + f * (b1 - b0), n0 + f * (n1 - n0))
    return None


if __name__ == "__main__":
    for k in ["wing_xtip", "wing_sweep", "wing_MAC", "wing_xRef", "fus_xcyl2", "fus_xtail",
              "og_xtip", "og_xT", "vt_xroot", "vt_xtip", "st_xt", "in_xlip", "in_ref", "in_rex", "in_yc0"]:
        print(f"{k:12s} {globals()[k]: .4f}")
