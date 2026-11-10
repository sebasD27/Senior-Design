"""Build the superDrone CAD assembly (airframe + internal systems) and export STEP for SolidWorks.

    python3 build_superdrone.py            -> superDrone_gear_down.step, superDrone_gear_up.step

Geometry of the outer mold line follows superDrone_inlet_tefix.csm exactly (build frame: x aft,
y up, z starboard). Everything is modelled in metres and scaled to millimetres on export.
"""
import math, json, sys, time
import cadquery as cq
from cadquery import Vector as V, Location, Color
from sd_params import *
from sd_geom import *
from sd_airframe import *

GEAR_DOWN_DEFAULT = True

def PITOT():
    return cyl_x(og_xtip - 0.30, og_xtip + 0.06, 0.008).fuse(cyl_x(og_xtip - 0.33, og_xtip - 0.30, 0.004))

# ---------------------------------------------------------------- layout (stations, m)
ST = dict(
    avi_fwd=-7.30, avi_aft=-5.48,            # avionics bay bulkheads
    nlg_bay=(-5.40, -4.50), nlg_pivot=(-4.65, -0.06, 0.0), nlg_L=0.61, nlg_rw=0.08, nlg_ww=0.04,
    tank1=(-4.30, -2.10), bh_t1f=-4.38, bh_t1a=-2.05,
    tank2=(-1.95, -0.12), tank2_r=0.26,
    eq_bay=(0.05, 0.85), eq_tray_y=0.03,
    wing_fs=0.15, wing_rs=0.65, fin_fs=0.20, fin_rs=0.65,
    mlg_bay=(0.78, 1.82), mlg_pivot=(1.62, -0.02, 0.24), mlg_L=0.64, mlg_rw=0.09, mlg_ww=0.05,
    eng_face=2.42, eng_end=3.33,
    ground_y=-0.75,
)
WING_ROOT_Z = 0.300           # first rib outboard of the fuselage skin
FIN_ROOT_H = 0.302            # first fin rib above the fuselage top

# ring frames (x, label); spar frames sit where the wing / fin spars meet the fuselage
def wing_spar_x(frac, z=WING_ROOT_Z): return wing_xle(z) + frac * wing_c(z)
def fin_spar_x(frac, h=FIN_ROOT_H): return fin_xle(h) + frac * fin_c(h)

RING_FRAMES = [
    (-3.20, "F3 tank-1 support"),
    (-1.00, "F5 inlet frame"),
    (-0.05, "F6 equipment-bay fwd"),
    (round(wing_spar_x(ST["wing_fs"]), 3), "F7 wing front-spar frame"),
    (round(wing_spar_x(ST["wing_rs"]), 3), "F8 wing rear-spar / MLG frame"),
    (round(fin_spar_x(ST["fin_fs"]), 3), "F9 fin front-spar / MLG trunnion frame"),
    (round(fin_spar_x(ST["fin_rs"]), 3), "F10 fin rear-spar frame"),
    (2.60, "F11 engine fwd-mount frame"),
    (3.20, "F12 engine aft-mount frame"),
]
BULKHEADS = [
    (ST["avi_fwd"], "B1 avionics fwd bulkhead"),
    (ST["avi_aft"], "B2 avionics aft / NLG bay bulkhead"),
    (ST["bh_t1f"], "B3 tank-1 fwd bulkhead"),
    (ST["bh_t1a"], "B4 tank-1 aft bulkhead"),
]

C = dict(
    skin=Color(0.80, 0.82, 0.85), door=Color(0.70, 0.73, 0.78), duct=Color(0.62, 0.64, 0.68),
    struct=Color(0.30, 0.45, 0.70), fitting=Color(0.20, 0.30, 0.50),
    tank=Color(0.95, 0.70, 0.20), fuel=Color(0.85, 0.35, 0.10),
    engine=Color(0.45, 0.45, 0.48), hot=Color(0.60, 0.35, 0.25), rotor=Color(0.75, 0.72, 0.60),
    gear=Color(0.35, 0.35, 0.38), tire=Color(0.10, 0.10, 0.10), hub=Color(0.65, 0.65, 0.65),
    act=Color(0.75, 0.60, 0.20),
    avionics=Color(0.20, 0.60, 0.35), battery=Color(0.15, 0.35, 0.25), probe=Color(0.85, 0.85, 0.20),
)

# ================================================================ part registry
class Parts:
    def __init__(self):
        self.items = []   # (group, name, shape, color)
    def add(self, group, name, shape, color):
        shape = shape.clean() if hasattr(shape, "clean") else shape
        if not shape.isValid():
            shape = shape.fix()
            if not shape.isValid(): print(f"  !! {name}: still invalid after fix")
        if shape is None or not shape.Solids():
            print(f"  !! {name}: empty, skipped"); return
        self.items.append((group, name, shape, color))
    def get(self, name):
        for g, n, s, c in self.items:
            if n == name: return s
        raise KeyError(name)


def log(msg, t0=[time.time()]):
    print(f"[{time.time()-t0[0]:6.1f}s] {msg}", flush=True)


# ================================================================ builders
def build(gear_down=True):
    P = Parts()
    log(f"building ({'gear down' if gear_down else 'gear up'})")

    # ---------------------------------------------------------- OML pieces
    oml = fuselage_oml()
    inner_fus = fuselage_inner()
    cowl = loft_secs(cowl_sections())
    cowl_in = loft_secs(cowl_sections(x_start=in_xlip + 0.015, shrink=T_COWL))
    duct_out_ext = loft_secs(duct_sections(in_xlip - in_lext, grow=T_DUCT))
    duct_flow = loft_secs(duct_sections(in_xlip - in_lext)).fuse(cyl_x(in_xface - 0.002, in_xface + 0.05, in_ref))
    duct_wall = duct_out_ext.cut(duct_flow).intersect(box(in_xlip, in_xface, -1, 1, -1, 1))
    duct_clear = loft_secs(duct_sections(in_xlip - in_lext, grow=0.03))  # keep-out for tanks
    wo_R = wing_outer_R(); wi_R = wing_inner_R()
    wo_L = wo_R.mirror("XY"); wi_L = wi_R.mirror("XY")
    fo = fin_up(fin_outer_local()); fi = fin_up(fin_inner_local())
    log("OML lofts done")

    # ---------------------------------------------------------- landing-gear bay openings
    mb0, mb1 = ST["mlg_bay"]
    mlg_box_R = box(mb0, mb1, -0.36, 0.0, 0.205, 0.40)
    mlg_box_L = mlg_box_R.mirror("XY")
    nb0, nb1 = ST["nlg_bay"]
    nlg_box_R = box(nb0, nb1, -0.40, -0.05, 0.0, 0.055)
    nlg_box_L = box(nb0, nb1, -0.40, -0.05, -0.055, 0.0)

    # ---------------------------------------------------------- longerons, fittings (needed for cuts)
    lon_pos = [(0.196, 0.196, "upper stbd"), (0.196, -0.196, "upper port"),
               (-0.205, 0.190, "lower stbd"), (-0.205, -0.190, "lower port")]
    x_lon0, x_lon1 = ST["bh_t1a"] + 0.004, 2.60 - FRAME_T / 2
    longerons = []
    xf6 = RING_FRAMES[2][0] - FRAME_T / 2
    xf8 = RING_FRAMES[4][0] + FRAME_T / 2
    for y, z, lbl in lon_pos:
        if y > 0:
            longerons.append((f"Longeron {lbl}", box(x_lon0, x_lon1, y - 0.01, y + 0.01, z - 0.01, z + 0.01)))
        else:   # the S-duct passes through the lower longeron line between F6 and F8
            longerons.append((f"Longeron {lbl} fwd", box(x_lon0, xf6, y - 0.01, y + 0.01, z - 0.01, z + 0.01)))
            longerons.append((f"Longeron {lbl} aft", box(xf8, x_lon1, y - 0.01, y + 0.01, z - 0.01, z + 0.01)))

    # wing root fittings (one per spar per side) and fin root fittings
    fittings = []
    for frac, lbl in [(ST["wing_fs"], "front"), (ST["wing_rs"], "rear")]:
        xs_ = wing_spar_x(frac)
        th = naca_yt(frac, wing_thickr) * wing_c(WING_ROOT_Z)
        hh = 0.55 * th
        fR = box(xs_ - 0.03, xs_ + 0.03, wing_yloc - hh, wing_yloc + hh, 0.262, WING_ROOT_Z - 0.0005)
        fittings.append((f"Wing root fitting {lbl} stbd", fR))
        fittings.append((f"Wing root fitting {lbl} port", fR.mirror("XY")))
    for frac, lbl in [(ST["fin_fs"], "front"), (ST["fin_rs"], "rear")]:
        xs_ = fin_spar_x(frac)
        th = naca_yt(frac * vt_tekeep, vt_thick) * fin_c(FIN_ROOT_H) / vt_tekeep
        hh = min(0.5 * th, 0.015)
        fittings.append((f"Fin root fitting {lbl}", box(xs_ - 0.025, xs_ + 0.025, 0.268, FIN_ROOT_H - 0.0005, -hh, hh)))

    # ---------------------------------------------------------- fuel lines (needed for frame cuts)
    yl, zl = 0.07, 0.10
    t2a = ST["tank2"][1]
    lines = {
        "Fuel line transfer T1-T2": [(ST["tank1"][1], 0.18, 0.05), (ST["tank2"][0], 0.18, 0.05)],
        "Fuel line T2-pump": [(t2a, yl, zl), (0.10, yl, zl)],
        "Fuel line pump-filter": [(0.24, yl, zl), (0.28, yl, zl)],
        "Fuel line filter-valve": [(0.34, yl, zl), (0.38, yl, zl)],
        "Fuel line valve-engine": [(0.44, yl, zl), (0.47, yl, zl), (0.53, 0.22, zl), (2.58, 0.22, zl), (2.725, 0.1639, zl)],
    }
    line_solids = {k: polyline_pipe(v, 0.006) for k, v in lines.items()}
    line_keepout = None
    for k, v in lines.items():
        ko = polyline_pipe(v, 0.009)
        line_keepout = ko if line_keepout is None else line_keepout.fuse(ko)

    # ---------------------------------------------------------- fuselage skin + doors
    outer = oml.fuse(cowl)
    cav = inner_fus.fuse(cowl_in)
    skin = outer.cut(cav).cut(duct_out_ext)
    for _, f in fittings:
        skin = skin.cut(f)
    skin = skin.cut(PITOT())
    skin = largest_solid(skin)
    log("skin booleans done")
    doors = {}
    for nm, bx in [("MLG door stbd", mlg_box_R), ("MLG door port", mlg_box_L),
                   ("NLG door stbd", nlg_box_R), ("NLG door port", nlg_box_L)]:
        doors[nm] = largest_solid(skin.intersect(bx))
    for bx in [mlg_box_R, mlg_box_L, nlg_box_R, nlg_box_L]:
        skin = skin.cut(bx)
    skin = largest_solid(skin)
    P.add("Airframe", "Fuselage skin + inlet cowl", skin, C["skin"])

    if gear_down:
        # main doors hinge on their upper edge (y = 0, side of fuselage), swing out 80 deg
        doors["MLG door stbd"] = rot_about(doors["MLG door stbd"], (0, 0, fus_radius), (1, 0, 0), -80)
        doors["MLG door port"] = rot_about(doors["MLG door port"], (0, 0, -fus_radius), (1, 0, 0), 80)
        xm = 0.5 * (nb0 + nb1); yh = -math.sqrt(r_oml(xm) ** 2 - 0.055 ** 2)
        doors["NLG door stbd"] = rot_about(doors["NLG door stbd"], (0, yh, 0.055), (1, 0, 0), -90)
        doors["NLG door port"] = rot_about(doors["NLG door port"], (0, yh, -0.055), (1, 0, 0), 90)
        # nudge each open door outboard until it clears the skin around its hinge
        for nm in doors:
            sgn = 1 if "stbd" in nm else -1
            for _ in range(12):
                if doors[nm].intersect(skin).Volume() < 1e-9: break
                doors[nm] = doors[nm].translate(V(0, -0.001, 0.0015 * sgn))
    for nm, d in doors.items():
        P.add("Airframe/Gear doors", nm, d, C["door"])

    # ---------------------------------------------------------- inlet duct, base ring
    P.add("Airframe", "Inlet S-duct", duct_wall, C["duct"])
    xb = fus_xtail - 0.006
    P.add("Airframe", "Aft base ring", tube_x(xb, fus_xtail, r_inner(xb) - 0.0005, in_rex + 0.010), C["struct"])

    # ---------------------------------------------------------- wings
    for side, wo, wi, mir in [("stbd", wo_R, wi_R, False), ("port", wo_L, wi_L, True)]:
        P.add("Airframe/Wings", f"Wing skin {side}", largest_solid(wo.cut(wi).cut(oml)), C["skin"])
    z1 = wing_span - 0.004
    spars = spar_set(wi_R, wing_xle, wing_c, WING_ROOT_Z, z1, [ST["wing_fs"], ST["wing_rs"]])
    ribs = rib_set(wi_R, [WING_ROOT_Z, 0.55, 0.80, 1.05, 1.27], spars)
    for side, mir in [("stbd", False), ("port", True)]:
        f = (lambda s: s.mirror("XY")) if mir else (lambda s: s)
        P.add("Structure/Wing " + side, f"Wing front spar {side}", f(spars[0]), C["struct"])
        P.add("Structure/Wing " + side, f"Wing rear spar {side}", f(spars[1]), C["struct"])
        for i, r in enumerate(ribs):
            P.add("Structure/Wing " + side, f"Wing rib {i+1} {side}", f(r), C["struct"])
    log("wings done")

    # ---------------------------------------------------------- fin
    P.add("Airframe/Fin", "Fin skin", largest_solid(fo.cut(fi).cut(oml)), C["skin"])
    fi_l = fin_inner_local()
    fsp = spar_set(fi_l, fin_xle, fin_c, FIN_ROOT_H, vt_height - 0.004, [ST["fin_fs"], ST["fin_rs"]])
    fri = rib_set(fi_l, [FIN_ROOT_H, 0.55, 0.80, 1.05], fsp)
    P.add("Structure/Fin", "Fin front spar", fin_up(fsp[0]), C["struct"])
    P.add("Structure/Fin", "Fin rear spar", fin_up(fsp[1]), C["struct"])
    for i, r in enumerate(fri):
        P.add("Structure/Fin", f"Fin rib {i+1}", fin_up(r), C["struct"])

    # ---------------------------------------------------------- strakes (LERX)
    st = strake_loft().cut(oml).cut(wo_R).cut(wo_L)
    for s in st.Solids():
        side = "stbd" if s.Center().z > 0 else "port"
        P.add("Airframe/Strakes", f"Strake (LERX) {side}", s, C["skin"])
    log("fin + strakes done")

    # ---------------------------------------------------------- fuselage frames / bulkheads / longerons
    for x, lbl in RING_FRAMES:
        ro = r_inner(x) - 0.0005
        fr = tube_x(x - FRAME_T / 2, x + FRAME_T / 2, ro, ro - FRAME_DEPTH)
        fr = fr.cut(duct_out_ext)
        for bx in [mlg_box_R, mlg_box_L]:
            fr = fr.cut(bx)
        for _, l in longerons: fr = fr.cut(l)
        for _, f in fittings: fr = fr.cut(f)
        fr = fr.cut(line_keepout)
        P.add("Structure/Fuselage", lbl, fr, C["struct"])
    for x, lbl in BULKHEADS:
        bh = cyl_x(x - 0.004, x + 0.004, r_inner(x) - 0.0005)
        # lightening hole in the avionics bulkheads, feed-through in the tank bulkheads
        if "avionics fwd" in lbl:
            bh = bh.cut(cyl_x(x - 0.01, x + 0.01, 0.05, 0.0, 0))
        bh = bh.cut(line_keepout)
        P.add("Structure/Fuselage", lbl, bh, C["struct"])
    for n, l in longerons:
        P.add("Structure/Fuselage", n, l.cut(line_keepout) if l.intersect(line_keepout).Volume() > 0 else l, C["struct"])
    for n, f in fittings:
        P.add("Structure/Airframe fittings", n, f, C["fitting"])
    log("fuselage structure done")

    # ---------------------------------------------------------- fuel system
    t1a, t1b = ST["tank1"]
    xr = [(t1a + i * (t1b - t1a) / 20, r_inner(t1a + i * (t1b - t1a) / 20) - 0.030) for i in range(21)]
    tank1 = revolve_spline_xr(xr)
    try:
        tank1 = cq.Workplane().add(tank1).edges().fillet(0.02).val()
    except Exception:
        pass
    P.add("Fuel system", "Fuel tank 1 (fwd, conformal)", tank1, C["tank"])
    t2 = cyl_x(ST["tank2"][0], ST["tank2"][1], ST["tank2_r"]).cut(duct_clear)
    for _, l in longerons: t2 = t2.cut(l)
    P.add("Fuel system", "Fuel tank 2 (saddle over duct)", largest_solid(t2), C["tank"])

    ty = ST["eq_tray_y"]
    e0, e1 = -0.05 + FRAME_T / 2, ST["eq_bay"][1]
    P.add("Structure/Equipment", "Equipment tray", box(e0, e1, ty, ty + 0.006, -0.18, 0.18), C["struct"])
    tt = ty + 0.006
    pump = cyl_x(0.10, 0.20, 0.025, yl, zl).fuse(box(0.20, 0.24, yl - 0.025, yl + 0.025, zl - 0.025, zl + 0.025))
    pump = pump.fuse(box(0.10, 0.24, tt, yl - 0.02, zl - 0.03, zl + 0.03))   # mounting foot
    P.add("Fuel system", "Fuel pump (brushless, gear type)", pump, C["fuel"])
    filt = cyl_x(0.28, 0.34, 0.018, yl, zl).fuse(box(0.29, 0.33, tt, yl - 0.012, zl - 0.015, zl + 0.015))
    P.add("Fuel system", "Fuel filter", filt, C["fuel"])
    P.add("Fuel system", "Fuel shut-off solenoid valve",
          box(0.38, 0.44, tt, yl + 0.02, zl - 0.02, zl + 0.02), C["fuel"])
    for k, s in line_solids.items():
        P.add("Fuel system/Lines", k, s, C["fuel"])
    log("fuel system done")

    # ---------------------------------------------------------- equipment bay electronics
    P.add("Avionics", "Engine ECU", box(0.50, 0.70, tt, tt + 0.05, -0.17, -0.03), C["avionics"])
    P.add("Avionics", "Main battery (engine / actuators)", box(0.10, 0.40, tt, tt + 0.08, -0.17, -0.03), C["battery"])
    P.add("Avionics", "Gear / servo controller", box(0.50, 0.62, tt, tt + 0.03, 0.02, 0.07), C["avionics"])

    # ---------------------------------------------------------- avionics bay (nose)
    a0, a1 = -6.95, ST["avi_aft"] - 0.004 - 0.06
    ay = -0.05
    tray = box(a0, a1, ay, ay + 0.006, -0.09, 0.09)
    P.add("Structure/Avionics bay", "Avionics tray", tray, C["struct"])
    for x0, x1 in [(a0 + 0.02, a0 + 0.04), (a1 - 0.04, a1 - 0.02)]:   # tray legs
        leg = box(x0, x1, -0.2, ay, -0.07, 0.07).intersect(cyl_x(x0 - 0.1, x1 + 0.1, r_inner(x0) - 0.001))
        P.add("Structure/Avionics bay", f"Avionics tray support x={x0:.2f}", leg, C["struct"])
    at = ay + 0.006
    boxes = [("Air data computer", -6.90, -6.80, 0.040, 0.040),
             ("Flight control computer", -6.75, -6.55, 0.050, 0.060),
             ("INS / GNSS unit", -6.48, -6.33, 0.060, 0.050),
             ("Datalink radio", -6.25, -6.05, 0.050, 0.070),
             ("Power distribution unit", -5.98, -5.85, 0.040, 0.060),
             ("Avionics battery", -5.80, a1 + 0.0, 0.070, 0.080)]
    for nm, x0, x1, h, hw in boxes:
        P.add("Avionics", nm, box(x0, x1, at, at + h, -hw, hw), C["battery"] if "battery" in nm else C["avionics"])
    P.add("Avionics", "Pitot-static probe", PITOT(), C["probe"])
    P.add("Avionics", "Pitot line to ADC", polyline_pipe([(og_xtip + 0.06, 0, 0), (-6.90, at + 0.02, 0)], 0.003), C["probe"])
    xg = -5.10; ytop = r_inner(xg)
    P.add("Avionics", "GNSS antenna (under skin)", box(xg - 0.03, xg + 0.03, ytop - 0.022, ytop - 0.012, -0.03, 0.03), C["avionics"])
    # datalink blade antenna on the spine
    bw = cq.Workplane("XY").polyline([(0.15, fus_radius), (0.27, fus_radius),
                                      (0.24, fus_radius + 0.09), (0.20, fus_radius + 0.09)]).close() \
        .extrude(0.004).translate((0, 0, -0.002)).val()
    P.add("Avionics", "Datalink blade antenna", bw, C["avionics"])
    log("avionics done")

    # ---------------------------------------------------------- propulsion
    build_turbojet(P)
    log("turbojet done")

    # ---------------------------------------------------------- landing gear
    build_gear(P, gear_down)
    log("landing gear done")
    return P


# ================================================================ turbojet
def build_turbojet(P):
    G = "Propulsion/Turbojet"
    x0 = ST["eng_face"]
    injectors = [rot_about(cyl((2.725, 0.0, 0.1765), (2.725, 0.0, 0.1595), 0.003), (0, 0, 0), (1, 0, 0), 45 * i + 22.5)
                 for i in range(8)]
    P.add(G, "Intake / compressor casing",
          revolve_xr([(x0, 0.172), (x0, 0.180), (2.700, 0.180), (2.700, 0.171), (2.66, 0.171), (2.62, 0.160),
                      (2.53, 0.110), (2.50, 0.172)]), C["engine"])
    bullet = revolve_spline_xr([(x0 + 0.01, 0.0005), (2.45, 0.022), (2.47, 0.035), (2.51, 0.052), (2.525, 0.055)])
    for i in range(3):
        s = box(2.46, 2.48, 0.025, 0.1715, -0.004, 0.004)
        bullet = bullet.fuse(rot_about(s, (0, 0, 0), (1, 0, 0), 120 * i))
    P.add(G, "Starter housing + intake struts", bullet, C["engine"])
    hub = revolve_xr([(2.525, 0.0), (2.525, 0.045), (2.565, 0.058), (2.600, 0.095), (2.620, 0.150),
                      (2.632, 0.155), (2.632, 0.0)])
    env = revolve_xr([(2.528, 0.040), (2.528, 0.105), (2.570, 0.128), (2.605, 0.147), (2.628, 0.154),
                      (2.628, 0.040)])
    blades = None
    for i in range(12):
        b = rot_about(box(2.52, 2.64, 0.0, 0.2, -0.002, 0.002), (0, 0, 0), (1, 0, 0), 30 * i)
        blades = b if blades is None else blades.fuse(b)
    P.add(G, "Compressor impeller (centrifugal)", hub.fuse(env.intersect(blades)), C["rotor"])
    P.add(G, "Diffuser ring", tube_x(2.662, 2.698, 0.1705, 0.158), C["engine"])
    cas = tube_x(2.700, 3.080, 0.175, 0.171)
    for j in injectors: cas = cas.cut(j)
    P.add(G, "Combustor outer casing", cas, C["hot"])
    P.add(G, "Combustor liner outer", tube_x(2.720, 3.040, 0.158, 0.155), C["hot"])
    P.add(G, "Combustor liner inner", tube_x(2.720, 3.040, 0.078, 0.075), C["hot"])
    P.add(G, "Combustor dome", tube_x(2.712, 2.720, 0.158, 0.075), C["hot"])
    P.add(G, "Shaft tunnel", tube_x(2.640, 3.060, 0.030, 0.026), C["engine"])
    P.add(G, "Main shaft", cyl_x(2.632, 3.105, 0.015), C["rotor"])
    P.add(G, "Front bearing", tube_x(2.645, 2.660, 0.026, 0.015), C["engine"])
    P.add(G, "Rear bearing", tube_x(3.040, 3.055, 0.026, 0.015), C["engine"])
    P.add(G, "Fuel manifold ring", torus_x(2.725, 0.181, 0.005), C["fuel"])
    for i, j in enumerate(injectors):
        P.add(G, f"Fuel injector {i+1}", j, C["fuel"])
    ngv = tube_x(3.080, 3.100, 0.105, 0.100).fuse(tube_x(3.080, 3.100, 0.160, 0.156))
    for i in range(16):
        v = box(3.082, 3.098, 0.104, 0.157, -0.0015, 0.0015).rotate(V(3.09, 0, 0), V(3.09, 1, 0), 35)
        ngv = ngv.fuse(rot_about(v, (0, 0, 0), (1, 0, 0), 22.5 * i))
    P.add(G, "Turbine NGV ring", ngv, C["hot"])
    wheel = cyl_x(3.105, 3.130, 0.100).fuse(cyl_x(3.130, 3.136, 0.04))
    for i in range(24):
        b = box(3.107, 3.128, 0.099, 0.152, -0.0015, 0.0015).rotate(V(3.1175, 0, 0), V(3.1175, 1, 0), -30)
        wheel = wheel.fuse(rot_about(b, (0, 0, 0), (1, 0, 0), 15 * i))
    P.add(G, "Turbine wheel", wheel, C["rotor"])
    P.add(G, "Exhaust casing",
          revolve_xr([(3.080, 0.168), (3.080, 0.172), (3.330, 0.165), (3.330, 0.161)]), C["hot"])
    cone = revolve_xr([(3.140, 0.0), (3.140, 0.100), (3.320, 0.030), (3.320, 0.0)])
    for i in range(4):
        s = box(3.20, 3.22, 0.060, 0.1640, -0.003, 0.003)
        cone = cone.fuse(rot_about(s, (0, 0, 0), (1, 0, 0), 45 + 90 * i))
    P.add(G, "Exhaust cone + struts", cone, C["hot"])
    P.add("Propulsion", "Tailpipe / convergent nozzle",
          revolve_xr([(3.330, 0.161), (3.330, 0.165), (fus_xtail + 0.03, in_rex + 0.002), (fus_xtail + 0.03, in_rex)]),
          C["hot"])
    P.add("Propulsion/Engine mounts", "Engine fwd mount ring", tube_x(2.585, 2.615, 0.190, 0.1801), C["fitting"])
    P.add("Propulsion/Engine mounts", "Engine aft mount ring", tube_x(3.185, 3.215, 0.178, 0.1701), C["fitting"])
    for i, ang in enumerate([0, 90, 180, 270]):
        rf = r_inner(2.60) - FRAME_DEPTH - 0.0008
        sf = rot_about(cyl((2.60, 0, 0.1905), (2.60, 0, rf), 0.010), (0, 0, 0), (1, 0, 0), ang)
        P.add("Propulsion/Engine mounts", f"Engine fwd mount strut {i+1}", sf, C["fitting"])
        ra = r_inner(3.20) - FRAME_DEPTH - 0.0008
        sa = rot_about(cyl((3.20, 0, 0.1785), (3.20, 0, ra), 0.008), (0, 0, 0), (1, 0, 0), ang)
        P.add("Propulsion/Engine mounts", f"Engine aft mount strut {i+1}", sa, C["fitting"])


# ================================================================ landing gear
def gear_leg_local(L, rw, ww, r_up, r_pis, head_r, pin_r, lug_xy):
    """Strut built at the pivot (origin), pointing down -y, wheel axle along z."""
    half = ww / 2
    arm0, arm1 = half + 0.002, half + 0.008
    pin = cyl((0, 0, -arm1 - 0.01), (0, 0, arm1 + 0.01), pin_r)
    head = cyl((0, 0, -half - 0.001), (0, 0, half + 0.001), head_r).cut(pin)
    upper = cyl((0, 0, 0), (0, -0.30, 0), r_up).cut(pin)
    sleeve = cyl((0, -0.30, 0), (0, -0.36, 0), r_up).cut(cyl((0, -0.29, 0), (0, -0.37, 0), r_pis))
    lx, ly = lug_xy
    tab = box(0, lx + 0.01, ly - 0.01, ly + 0.01, -0.006, 0.006).cut(cyl((lx, ly, -0.01), (lx, ly, 0.01), 0.004))
    cylinder = head.fuse(upper).fuse(sleeve).fuse(tab)
    y_cb = -(L - rw - 0.015)            # crown bottom, clear of the tyre
    y_ct = y_cb + 0.03
    piston = cyl((0, -0.32, 0), (0, y_ct, 0), r_pis)
    crown = box(-0.02, 0.02, y_cb, y_ct, -arm1, arm1)
    arms = box(-0.015, 0.015, -L - 0.02, y_cb, arm0, arm1).fuse(box(-0.015, 0.015, -L - 0.02, y_cb, -arm1, -arm0))
    axle = cyl((0, -L, -arm1), (0, -L, arm1), 0.010)
    fork = crown.fuse(arms).cut(axle)
    tire = cyl((0, -L, -half), (0, -L, half), rw).cut(cyl((0, -L, -half - 0.01), (0, -L, half + 0.01), 0.66 * rw))
    es = [e for e in tire.Edges() if e.geomType() == "CIRCLE" and abs(e.radius() - rw) < 1e-6]
    try:
        tire = tire.fillet(0.4 * half, es)
    except Exception as ex:
        print("   tyre fillet skipped", ex)
    hub = cyl((0, -L, -half + 0.002), (0, -L, half - 0.002), 0.66 * rw).cut(cyl((0, -L, -half), (0, -L, half), 0.0101))
    return dict(cylinder=cylinder, piston=piston, fork=fork, axle=axle, tire=tire, hub=hub, pin=pin)


def place(shape, pivot, ang):
    return shape.rotate(V(0, 0, 0), V(0, 0, 1), ang).translate(V(*pivot))


def actuator(A, P_, Lb, Lr, rb, rr):
    A, P_ = V(*A), V(*P_)
    d = (P_ - A).normalized()
    T = lambda v: v.toTuple()
    barrel = cyl(T(A), T(A + d * Lb), rb).fuse(sphere(T(A), rb * 0.9))
    barrel = barrel.cut(cyl(T(A + d * 0.004), T(A + d * (Lb + 0.01)), rr + 0.0005))
    rod = cyl(T(P_ - d * 0.017), T(P_ - d * Lr), rr)
    L = (A - P_).Length
    assert L - Lr > 0.006, f"actuator bottoms out ({L:.3f} vs rod {Lr})"
    assert Lb - (L - Lr) > 0.025, f"actuator rod leaves barrel ({L:.3f})"
    return barrel, rod, L


def envelope(A, P_, rb):
    A, P_ = V(*A), V(*P_)
    d = (P_ - A).normalized()
    return cyl((A - d * 0.03).toTuple(), (A + d * 0.3).toTuple(), rb + 0.0005).fuse(sphere(A.toTuple(), rb + 0.0005))


def build_gear(P, gear_down):
    ang = 0.0 if gear_down else -90.0           # retract forward about the trunnion (z) axis
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    # ---------------- nose gear
    px, py, pz = ST["nlg_pivot"]
    lx, ly = 0.03, -0.25
    leg = gear_leg_local(ST["nlg_L"], ST["nlg_rw"], ST["nlg_ww"], 0.020, 0.013, 0.025, 0.009, (lx, ly))
    G = "Landing gear/Nose gear"
    names = dict(cylinder="NLG oleo cylinder", piston="NLG oleo piston", fork="NLG fork", axle="NLG axle",
                 tire="NLG tyre", hub="NLG wheel hub", pin="NLG trunnion pin")
    for k, nm in names.items():
        col = C["tire"] if k == "tire" else C["hub"] if k == "hub" else C["gear"]
        P.add(G, nm, place(leg[k], (px, py, pz), ang), col)
    blk = box(px + 0.01, ST["bh_t1f"] - 0.004, -0.025, 0.02, -0.03, 0.03)
    lugs = box(px - 0.035, px + 0.035, py - 0.03, -0.02, 0.022, 0.030).fuse(
        box(px - 0.035, px + 0.035, py - 0.03, -0.02, -0.030, -0.022))
    P.add("Landing gear/Gear fittings", "NLG trunnion fitting", blk.fuse(lugs).cut(place(leg["pin"], (px, py, pz), 0)), C["fitting"])
    A = (-4.40, -0.12, 0.0)
    Pp = (px + lx * c - ly * s, py + lx * s + ly * c, 0.0)
    brk = box(A[0] - 0.015, ST["bh_t1f"] - 0.004, -0.14, -0.10, -0.02, 0.02)
    bar, rod, Lap = actuator(A, Pp, 0.25, 0.28, 0.012, 0.006)
    P.add("Landing gear/Gear fittings", "NLG actuator bracket", brk.cut(envelope(A, Pp, 0.012)), C["fitting"])
    P.add("Landing gear/Actuators", "NLG retraction actuator barrel", bar, C["act"])
    P.add("Landing gear/Actuators", "NLG retraction actuator rod", rod, C["hub"])
    print(f"   NLG actuator pin-to-pin {Lap:.3f} m")

    # ---------------- main gear (stbd, mirrored to port)
    px, py, pz = ST["mlg_pivot"]
    lx, ly = 0.04, -0.18
    leg = gear_leg_local(ST["mlg_L"], ST["mlg_rw"], ST["mlg_ww"], 0.025, 0.017, 0.030, 0.012, (lx, ly))
    A = (1.95, -0.10, pz)
    Pp = (px + lx * c - ly * s, py + lx * s + ly * c, pz)
    xf9 = RING_FRAMES[5][0]
    blk = box(px - 0.04, xf9 - FRAME_T / 2, 0.015, 0.06, 0.205, 0.275)
    lugs = box(px - 0.035, px + 0.035, py - 0.025, 0.02, 0.205, 0.213).fuse(
        box(px - 0.035, px + 0.035, py - 0.025, 0.02, 0.267, 0.275))
    fitR = blk.fuse(lugs).cut(place(leg["pin"], (px, py, pz), 0))
    bar, rod, Lap = actuator(A, Pp, 0.25, 0.295, 0.015, 0.008)
    brkR = box(1.935, 1.965, -0.215, -0.085, 0.200, 0.262).intersect(cyl_x(1.9, 2.0, r_inner(1.95) - 0.001)).cut(envelope(A, Pp, 0.015))
    print(f"   MLG actuator pin-to-pin {Lap:.3f} m")
    for side, mir in [("stbd", False), ("port", True)]:
        f = (lambda q: q.mirror("XY")) if mir else (lambda q: q)
        G = f"Landing gear/Main gear {side}"
        names = dict(cylinder="MLG oleo cylinder", piston="MLG oleo piston", fork="MLG fork", axle="MLG axle",
                     tire="MLG tyre", hub="MLG wheel hub", pin="MLG trunnion pin")
        for k, nm in names.items():
            col = C["tire"] if k == "tire" else C["hub"] if k == "hub" else C["gear"]
            P.add(G, f"{nm} {side}", f(place(leg[k], (px, py, pz), ang)), col)
        P.add("Landing gear/Gear fittings", f"MLG trunnion fitting {side}", f(fitR), C["fitting"])
        P.add("Landing gear/Gear fittings", f"MLG actuator bracket {side}", f(brkR), C["fitting"])
        P.add("Landing gear/Actuators", f"MLG retraction actuator barrel {side}", f(bar), C["act"])
        P.add("Landing gear/Actuators", f"MLG retraction actuator rod {side}", f(rod), C["hub"])


# ================================================================ export
def to_assembly(P, name, scale=1000.0):
    tree = {}
    for g, n, s, col in P.items:
        tree.setdefault(g, []).append((n, s, col))
    paths = set()
    for g in tree:
        segs = g.split("/")
        for i in range(1, len(segs) + 1):
            paths.add("/".join(segs[:i]))
    order = []
    for g, n, s, col in P.items:
        segs = g.split("/")
        for i in range(1, len(segs) + 1):
            p = "/".join(segs[:i])
            if p not in order: order.append(p)

    def make(path):
        a = cq.Assembly(name=path.split("/")[-1])
        for n, s, col in tree.get(path, []):
            a.add(s.scale(scale), name=n, color=col)
        for ch in [p for p in order if "/" in p and p.rsplit("/", 1)[0] == path]:
            a.add(make(ch), name=ch.split("/")[-1])
        return a

    root = cq.Assembly(name=name)
    for top in [p for p in order if "/" not in p]:
        root.add(make(top), name=top)
    return root


if __name__ == "__main__":
    configs = sys.argv[1:] or ["down", "up"]
    for cfg in configs:
        P = build(gear_down=(cfg == "down"))
        print(f"{len(P.items)} parts")
        import pickle
        pickle.dump([(g, n, s, (col.toTuple() if hasattr(col, "toTuple") else None)) for g, n, s, col in P.items],
                    open(f"parts_{cfg}.pkl", "wb"))
        assy = to_assembly(P, f"superDrone_gear_{cfg}")
        out = f"superDrone_gear_{cfg}.step"
        assy.save(out, exportType="STEP")
        log(f"wrote {out}")
