"""Offscreen VTK render of the built superDrone parts (from parts_<cfg>.pkl) with labelled call-outs."""
import pickle, sys, math
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

def poly_from_shape(s, tol=0.002):
    vs, tris = s.tessellate(tol, 0.3)
    if not tris:
        return None
    pts = np.array([v.toTuple() for v in vs], dtype=float)
    tri = np.array(tris, dtype=np.int64)
    cells = np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts, deep=True)); pd.SetPoints(vp)
    ca = vtk.vtkCellArray(); ca.SetCells(len(tri), numpy_to_vtkIdTypeArray(cells, deep=True)); pd.SetPolys(ca)
    n = vtk.vtkPolyDataNormals(); n.SetInputData(pd); n.SetFeatureAngle(35); n.Update()
    return n.GetOutput()

def is_skin(g, n):
    return n.startswith("Fuselage skin") or n.startswith("Wing skin") or n.startswith("Fin skin") \
        or n.startswith("Strake") or "door" in n

def render(items, out, cam, size=(2400, 1100), skin_opacity=0.10, labels=(), title="", hide=lambda g, n: False,
           parallel=None):
    ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(*size)
    win.SetMultiSamples(8)
    for g, n, s, col in items:
        if hide(g, n):
            continue
        pd = poly_from_shape(s, 0.004 if is_skin(g, n) else 0.0015)
        if pd is None:
            continue
        m = vtk.vtkPolyDataMapper(); m.SetInputData(pd)
        a = vtk.vtkActor(); a.SetMapper(m)
        p = a.GetProperty()
        rgb = col[:3] if col else (0.7, 0.7, 0.7)
        p.SetColor(*rgb); p.SetSpecular(0.25); p.SetSpecularPower(20)
        if is_skin(g, n):
            p.SetOpacity(skin_opacity)
            sil = vtk.vtkPolyDataSilhouette(); sil.SetInputData(pd); sil.SetCamera(ren.GetActiveCamera())
            sil.SetEnableFeatureAngle(0); sil.BorderEdgesOn()
            sm = vtk.vtkPolyDataMapper(); sm.SetInputConnection(sil.GetOutputPort())
            sa = vtk.vtkActor(); sa.SetMapper(sm)
            sa.GetProperty().SetColor(0.35, 0.38, 0.42); sa.GetProperty().SetLineWidth(1.6)
            ren.AddActor(sa)
        ren.AddActor(a)
    ren.SetUseDepthPeeling(1); ren.SetMaximumNumberOfPeels(8); win.SetAlphaBitPlanes(1)
    c = ren.GetActiveCamera()
    c.SetPosition(*cam["pos"]); c.SetFocalPoint(*cam["focal"]); c.SetViewUp(*cam["up"])
    if parallel:
        c.ParallelProjectionOn(); c.SetParallelScale(parallel)
    else:
        c.SetViewAngle(cam.get("angle", 22))
    ren.ResetCameraClippingRange()
    lk = vtk.vtkLight(); lk.SetLightTypeToCameraLight(); lk.SetPosition(0.3, 1, 1); lk.SetIntensity(0.55)
    ren.AddLight(lk)
    win.Render()
    # project label anchors
    coord = vtk.vtkCoordinate(); coord.SetCoordinateSystemToWorld()
    anchors = []
    for txt, p3, off in labels:
        coord.SetValue(*p3)
        x, y = coord.GetComputedDoubleDisplayValue(ren)
        anchors.append((txt, x, size[1] - y, off))
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.ReadFrontBufferOff(); w2i.Update()
    png = vtk.vtkPNGWriter(); png.SetFileName(out + ".raw.png"); png.SetInputConnection(w2i.GetOutputPort()); png.Write()
    annotate(out + ".raw.png", out, anchors, title, size)

def annotate(raw, out, anchors, title, size):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(raw)
    dpi = 200
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    for i, (txt, x, y, off) in enumerate(anchors):
        dx, dy = off
        ax.annotate(txt, xy=(x, y), xytext=(x + dx, y + dy), fontsize=6.5, color="#111",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444", lw=0.5, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#333", shrinkA=0, shrinkB=1))
        ax.plot([x], [y], "o", ms=2.2, color="#111")
    if title:
        ax.text(20, 40, title, fontsize=10, weight="bold", color="#111", va="top")
    fig.savefig(out, dpi=dpi); plt.close(fig)

if __name__ == "__main__":
    items_dn = pickle.load(open("parts_down.pkl", "rb"))
    items_up = pickle.load(open("parts_up.pkl", "rb"))
    # 1. full-length side cutaway, gear down (looking from starboard, nose left)
    side = dict(pos=(-2.25, 0.25, 30), focal=(-2.25, 0.25, 0), up=(0, 1, 0))
    L = [
        ("Pitot probe", (-8.0, 0, 0), (40, -150)),
        ("Avionics bay\nADC · FCC · INS/GNSS ·\ndatalink · PDU · battery", (-6.4, 0.0, 0), (0, -260)),
        ("Nose gear\n(retracts fwd)", (-4.65, -0.40, 0), (-150, 90)),
        ("Fuel tank 1\n(conformal)", (-3.2, 0.10, 0), (0, -230)),
        ("Fuel tank 2\n(saddle over duct)", (-1.0, 0.15, 0), (-40, -240)),
        ("Inlet lip", (-1.6, -0.45, 0), (-60, 110)),
        ("S-duct", (0.4, -0.25, 0), (-120, 130)),
        ("Equipment bay\nfuel pump · filter · valve ·\nECU · battery", (0.35, 0.10, 0), (60, -300)),
        ("Main gear\n(retracts fwd)", (1.62, -0.45, 0.24), (-70, 110)),
        ("Wing spar frames", (1.48, 0.25, 0.0), (160, -230)),
        ("Turbojet", (2.85, 0.0, 0), (-140, 190)),
        ("Tailpipe", (3.55, 0.0, 0), (-60, 170)),
        ("Fin spars + ribs", (2.1, 0.75, 0), (130, -60)),
    ]
    render(items_dn, "preview_side_gear_down.png", side, size=(2600, 900), skin_opacity=0.08, labels=L,
           title="superDrone - internal layout, gear down (starboard view, skins transparent)", parallel=2.35)
    # 2. isometric
    iso = dict(pos=(-9.5, 5.5, 9.0), focal=(-1.6, -0.1, 0), up=(0, 1, 0), angle=30)
    render(items_dn, "preview_iso_gear_down.png", iso, size=(2400, 1300), skin_opacity=0.12,
           title="superDrone - isometric, gear down")
    # 3. gear bay close-ups: up vs down
    det = dict(pos=(1.25, -0.28, 6.0), focal=(1.25, -0.28, 0), up=(0, 1, 0))
    hide = lambda g, n: n.startswith("Wing skin") or n.startswith("Strake")
    render(items_dn, "preview_mlg_down.png", det, size=(1600, 900), skin_opacity=0.10, hide=hide,
           title="Main gear - deployed (doors open)", parallel=0.6)
    render(items_up, "preview_mlg_up.png", det, size=(1600, 900), skin_opacity=0.10, hide=hide,
           title="Main gear - retracted into fuselage side bay", parallel=0.6)
    detn = dict(pos=(-4.9, -0.30, 6.0), focal=(-4.9, -0.30, 0), up=(0, 1, 0))
    render(items_dn, "preview_nlg_down.png", detn, size=(1200, 900), skin_opacity=0.10,
           title="Nose gear - deployed", parallel=0.55)
    render(items_up, "preview_nlg_up.png", detn, size=(1200, 900), skin_opacity=0.10,
           title="Nose gear - retracted", parallel=0.55)
    print("rendered")
