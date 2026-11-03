# The countries of the globe (web/globe.js): their outlines, simplified, from Natural Earth (public domain).
# Run: curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson
#      python3 web/globe_prep.py ne_50m_admin_0_countries.geojson web/globe.json
# Out: {"countries": [{"c": ISO code or "", "n": name, "x": lon, "y": lat (the place of the label), "p": rings}]}; a ring
# is a flat list of integers: lon and lat in tenths of a degree (11 km: less than a pixel of the globe).
import json
import sys

TOLERANCE = 0.12  # degrees: a point closer than this to the line of its neighbours goes
SMALL = 0.08  # square degrees: a smaller island goes, unless it is the largest part of its country


def simplify(points, tol):
    """Douglas-Peucker on a closed ring (the first point is the last)."""
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    todo = [(0, len(points) - 1)]
    while todo:
        a, b = todo.pop()
        (xa, ya), (xb, yb) = points[a], points[b]
        dx, dy = xb - xa, yb - ya
        norm = (dx * dx + dy * dy) ** 0.5
        far, best = -1.0, -1
        for i in range(a + 1, b):
            x, y = points[i]
            d = abs(dy * (x - xa) - dx * (y - ya)) / norm if norm else ((x - xa) ** 2 + (y - ya) ** 2) ** 0.5
            if d > far:
                far, best = d, i
        if far > tol:
            keep[best] = True
            todo += [(a, best), (best, b)]
    return [p for p, k in zip(points, keep) if k]


def area(points):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:]))) / 2


def main(src, out):
    countries = []
    for f in json.load(open(src))["features"]:
        p, geom = f["properties"], f["geometry"]
        code = p["ISO_A2_EH"] if p["ISO_A2_EH"] != "-99" else ""
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [ring for polygon in polygons for ring in polygon[:1]]  # the outer rings (the holes are lakes)
        largest = max(area(r) for r in rings)
        kept = []
        for ring in rings:
            if area(ring) < SMALL and area(ring) < largest:
                continue
            ring = simplify(ring, TOLERANCE)
            if len(ring) < 4:
                continue
            kept.append([v for x, y in ring for v in (round(x * 10), round(y * 10))])
        if not kept:  # a small country: its largest ring, not simplified
            ring = max(rings, key=area)
            kept = [[v for x, y in ring for v in (round(x * 10), round(y * 10))]]
        countries.append(
            {"c": code, "n": p["NAME"], "x": round(p["LABEL_X"], 2), "y": round(p["LABEL_Y"], 2), "p": kept}
        )
    json.dump(
        {"source": "Natural Earth 1:50m admin 0 countries (public domain), simplified", "countries": countries},
        open(out, "w"),
        separators=(",", ":"),
    )
    points = sum(len(r) // 2 for c in countries for r in c["p"])
    print(f"{out}: {len(countries)} countries, {points} points")


if __name__ == "__main__":
    main(*sys.argv[1:3])
