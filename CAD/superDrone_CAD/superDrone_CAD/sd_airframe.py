"""Outer mold line + primary structure of the superDrone (metres, build frame)."""
import math
import cadquery as cq
from cadquery import Vector as V
from sd_params import *
from sd_geom import *

T_FUS = 0.003     # fuselage skin
T_COWL = 0.004    # cowl skin
T_DUCT = 0.010    # inlet duct wall
T_WING = 0.002    # wing / fin skin
FRAME_DEPTH = 0.025
FRAME_T = 0.010


# ------------------------------------------------------------------ fuselage
def fuselage_oml():
    thm1 = (180 + og_thT) / 2
    thm2 = (og_thT + 90) / 2
    xm1, ym1 = og_xs + fus_rnose * cosd(thm1), fus_rnose * sind(thm1)
    xm2, ym2 = fus_xcyl1 + og_rho * cosd(thm2), og_cy + og_rho * sind(thm2)
    wp = (cq.Workplane("XY").moveTo(og_xtip, 0)
          .threePointArc((xm1, ym1), (og_xT, og_yT))
          .threePointArc((xm2, ym2), (fus_xcyl1, fus_radius))
          .lineTo(fus_xcyl2, fus_radius).lineTo(fus_xtail, fus_rtail)
          .lineTo(fus_xtail, 0).close())
    return wp.revolve(360, (0, 0, 0), (1, 0, 0)).val()


def fuselage_inner(t=T_FUS, x_ext=0.08):
    """Exact inward offset of the OML (concentric arcs), open past the base."""
    rn, rho = fus_rnose - t, og_rho - t
    thm1 = (180 + og_thT) / 2
    thm2 = (og_thT + 90) / 2
    p_tip = (og_xs - rn, 0)
    p_m1 = (og_xs + rn * cosd(thm1), rn * sind(thm1))
    p_T = (og_xs + rn * cosd(og_thT), rn * sind(og_thT))
    p_m2 = (fus_xcyl1 + rho * cosd(thm2), og_cy + rho * sind(thm2))
    p_c1 = (fus_xcyl1, fus_radius - t)
    # boattail line offset inward
    m = (fus_rtail - fus_radius) / fus_ltail
    s = math.sqrt(1 + m * m)
    # inner boattail: r = R + m (x - xcyl2) - t*s
    x_join = fus_xcyl2  # cylinder inner r = R - t ; boattail inner meets it at x where m(x-xcyl2) = -t*s + t
    x_join = fus_xcyl2 + (t - t * s) / m
    r_end = fus_radius + m * (fus_xtail + x_ext - fus_xcyl2) - t * s
    wp = (cq.Workplane("XY").moveTo(*p_tip)
          .threePointArc(p_m1, p_T).threePointArc(p_m2, p_c1)
          .lineTo(x_join, fus_radius - t).lineTo(fus_xtail + x_ext, r_end)
          .lineTo(fus_xtail + x_ext, 0).close())
    return wp.revolve(360, (0, 0, 0), (1, 0, 0)).val()


def r_inner(x, t=T_FUS):
    return max(r_oml(x) - t, 0.0)


def loft_secs(secs):
    return loft([supell_wire(*s) for s in secs])


# ------------------------------------------------------------------ wing / fin / strake lofts
def wing_outer_R():
    w0 = airfoil_wire(wing_thickr, wing_croot, wing_xroot, wing_yloc, 0.0)
    w1 = airfoil_wire(wing_thickt, wing_ctip, wing_xtip, wing_ytip, wing_span)
    return loft([w0, w1], ruled=True)


def wing_inner_R(t=T_WING):
    z1 = wing_span - 0.004
    w0 = airfoil_wire(wing_thickr, wing_croot, wing_xroot, wing_yloc, 0.0, inset=t)
    w1 = airfoil_wire(wing_thickt, wing_c(z1), wing_xle(z1), wing_yloc, z1, inset=t)
    return loft([w0, w1], ruled=True)


def fin_outer_local():
    w0 = airfoil_wire(vt_thick, vt_croot, vt_xroot, 0.0, 0.0, keep=vt_tekeep)
    w1 = airfoil_wire(vt_thick, vt_ctip, vt_xtip, 0.0, vt_height, keep=vt_tekeep)
    return loft([w0, w1], ruled=True)


def fin_inner_local(t=T_WING):
    h1 = vt_height - 0.004
    w0 = airfoil_wire(vt_thick, vt_croot, vt_xroot, 0.0, 0.0, keep=vt_tekeep, inset=t)
    w1 = airfoil_wire(vt_thick, fin_c(h1), fin_xle(h1), 0.0, h1, keep=vt_tekeep, inset=t)
    return loft([w0, w1], ruled=True)


def fin_up(shape):
    """Stand a fin built on +z up to +y (ROTATEX -90 in the .csm)."""
    return shape.rotate(V(0, 0, 0), V(1, 0, 0), -90)


def strake_loft():
    wa = airfoil_wire(st_thick, st_ctip, st_xt, st_yt, -st_zs)
    wb = airfoil_wire(st_thick, st_cr, wing_xroot - st_lfwd, wing_yloc, 0.0)
    wc = airfoil_wire(st_thick, st_ctip, st_xt, st_yt, st_zs)
    return loft([wa, wb, wc], ruled=True)


# ------------------------------------------------------------------ internal structure of a lifting surface
def spar_set(inner, xle_f, c_f, z0, z1, fracs, th=0.004):
    spars = []
    for f in fracs:
        xa = xle_f(z0) + f * c_f(z0)
        xb = xle_f(z1) + f * c_f(z1)
        slab = slab_between(xa - th / 2, xa + th / 2, z0, xb - th / 2, xb + th / 2, z1)
        spars.append(slab.intersect(inner).clean())
    return spars


def rib_set(inner, z_list, spars, th=0.004):
    ribs = []
    for z in z_list:
        r = box(-20, 20, -2, 2, z, z + th).intersect(inner)
        for s in spars:
            r = r.cut(s)
        ribs.append(r.clean())
    return ribs
