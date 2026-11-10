"""Low-level OpenCASCADE / CadQuery helpers for the superDrone model (metres, build frame)."""
import math
import cadquery as cq
from cadquery import Vector as V, Edge, Wire, Solid, Face, Compound

# ------------------------------------------------------------------ primitives
def box(x0, x1, y0, y1, z0, z1):
    return Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))

def cyl(p0, p1, r):
    p0, p1 = V(*p0), V(*p1)
    d = p1 - p0
    return Solid.makeCylinder(r, d.Length, p0, d.normalized())

def sphere(p, r):
    return Solid.makeSphere(r, V(*p), angleDegrees1=-90, angleDegrees2=90)

def cyl_x(x0, x1, r, y=0.0, z=0.0):
    return cyl((x0, y, z), (x1, y, z), r)

def tube_x(x0, x1, ro, ri, y=0.0, z=0.0):
    return cyl_x(x0, x1, ro, y, z).cut(cyl_x(x0 - 1e-3, x1 + 1e-3, ri, y, z))

def revolve_xr(pts):
    """Revolve a closed polygon given in (x, r) about the x axis."""
    return cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0)).val()

def revolve_spline_xr(xr_outer, x_close_axis=True):
    """Revolve a profile: spline through (x, r) points then back along the axis."""
    wp = cq.Workplane("XY").moveTo(xr_outer[0][0], 0).lineTo(*xr_outer[0])
    wp = wp.spline(xr_outer[1:], includeCurrent=True)
    wp = wp.lineTo(xr_outer[-1][0], 0).close()
    return wp.revolve(360, (0, 0, 0), (1, 0, 0)).val()

def torus_x(x, R, r):
    t = Solid.makeTorus(R, r, V(x, 0, 0), V(1, 0, 0))
    return t

def polyline_pipe(pts, r):
    """Round pipe through straight segments, joints filled with spheres (one solid)."""
    s = None
    for a, b in zip(pts[:-1], pts[1:]):
        c = cyl(a, b, r)
        s = c if s is None else s.fuse(c)
    for p in pts[1:-1]:
        s = s.fuse(sphere(p, r))
    return s.clean()

def rot_about(shape, p0, axis, ang):
    return shape.rotate(V(*p0), V(*p0) + V(*axis), ang)

def largest_solid(shape):
    sols = shape.Solids()
    return max(sols, key=lambda s: s.Volume()) if sols else shape

def as_solid(shape):
    sols = shape.Solids()
    if len(sols) == 1:
        return sols[0]
    return Compound.makeCompound(sols)

# ------------------------------------------------------------------ section wires
def supell_pts(x, yc, a, b, n, npts=96):
    pts = []
    for i in range(npts):
        th = 2 * math.pi * i / npts
        c, s = math.cos(th), math.sin(th)
        z = a * math.copysign(abs(c) ** (2.0 / n), c)
        y = yc + b * math.copysign(abs(s) ** (2.0 / n), s)
        pts.append(V(x, y, z))
    return pts

def supell_wire(x, yc, a, b, n, npts=96):
    e = Edge.makeSpline(supell_pts(x, yc, a, b, n, npts), periodic=True)
    return Wire.assembleEdges([e])

def loft(wires, ruled=False):
    return Solid.makeLoft(wires, ruled)

def naca_yt(x, t):
    return 5 * t * (0.2969 * math.sqrt(max(x, 0)) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1015 * x ** 4)

def airfoil_xy(t, chord, keep=1.0, inset=0.0, n=60):
    """Closed symmetric NACA 00xx loop (TE upper -> LE -> TE lower) in chord units scaled to `chord`.
    keep < 1 truncates the unit section at x = keep and rescales it so the chord stays `chord`
    (the .csm tekeep trick).  inset > 0 returns the inner skin line (offset inward by ~inset)."""
    S = chord / keep
    xs = [keep * (1 - math.cos(math.pi * i / n)) / 2 for i in range(n + 1)]
    pts = [(x * S, naca_yt(x, t) * S) for x in xs]
    if inset > 0:
        xmax = keep * S - inset
        inner = [(X, Y - inset) for X, Y in pts if Y - inset > 0.4 * inset and X <= xmax]
        # leading point on the camber line where the skin closes
        xle = inner[0][0] - 0.6 * inset
        up = [(X, Y) for X, Y in inner]
        loop = [(X, Y) for X, Y in reversed(up)] + [(xle, 0.0)] + [(X, -Y) for X, Y in up]
        return loop
    up = pts
    return [(X, Y) for X, Y in reversed(up)] + [(X, -Y) for X, Y in up[1:]]

def airfoil_wire(t, chord, xle, yle, z, keep=1.0, inset=0.0, n=60):
    loop = airfoil_xy(t, chord, keep, inset, n)
    pts = [V(xle + X, yle + Y, z) for X, Y in loop]
    sp = Edge.makeSpline(pts)
    te = Edge.makeLine(pts[-1], pts[0])
    return Wire.assembleEdges([sp, te])

def slab_between(x0a, x0b, z0, x1a, x1b, z1, ylo=-2.0, yhi=2.0):
    """Planar slab (spar web) bounded by the quad (x0a,z0)-(x0b,z0)-(x1b,z1)-(x1a,z1), extruded in y."""
    w = Wire.makePolygon([V(x0a, ylo, z0), V(x0b, ylo, z0), V(x1b, ylo, z1), V(x1a, ylo, z1)], close=True)
    return Solid.extrudeLinear(Face.makeFromWires(w), V(0, yhi - ylo, 0))
