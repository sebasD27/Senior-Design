"""Validity + pairwise interference check of a built configuration (volumes in cm^3)."""
import pickle, sys, time, itertools
sys.path.insert(0, '.')
cfg = sys.argv[1] if len(sys.argv) > 1 else "down"
items = pickle.load(open(f"parts_{cfg}.pkl", "rb"))
bad = [(n, s.isValid()) for g, n, s, c in items if not s.isValid()]
print("invalid solids:", bad)
bbs = [s.BoundingBox() for g, n, s, c in items]
def ov(a, b, tol=1e-4):
    return not (a.xmax < b.xmin - tol or b.xmax < a.xmin - tol or a.ymax < b.ymin - tol or b.ymax < a.ymin - tol
                or a.zmax < b.zmin - tol or b.zmax < a.zmin - tol)
pairs = [(i, j) for i, j in itertools.combinations(range(len(items)), 2) if ov(bbs[i], bbs[j])]
print(len(pairs), "bbox-overlapping pairs")
t = time.time(); hits = []
for i, j in pairs:
    try:
        v = items[i][2].intersect(items[j][2]).Volume()
    except Exception as e:
        v = -1
    if v > 1e-8 or v < 0:
        hits.append((v * 1e6, items[i][1], items[j][1]))
hits.sort(reverse=True)
print(f"{len(hits)} interferences ({time.time()-t:.0f}s)")
for v, a, b in hits:
    print(f"  {v:10.3f} cm3   {a}  <->  {b}")
