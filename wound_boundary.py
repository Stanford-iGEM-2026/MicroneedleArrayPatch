import json
import math
 
 
def load_boundary(path):
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("units") != "mm":
        raise ValueError(f"Expected units 'mm' in {path}, got {data.get('units')!r}")
    polygon = [(v["x"], v["y"]) for v in data["polygon"]]
    if len(polygon) < 3:
        raise ValueError(f"Boundary polygon in {path} has fewer than 3 vertices.")
    inset = None
    if data.get("inset_polygon"):
        inset = [(v["x"], v["y"]) for v in data["inset_polygon"]]
    return {
        "polygon": polygon,
        "inset_polygon": inset,
        "inset_margin_mm": data.get("inset_margin_mm"),
        "source": data.get("source", {}),
    }
 
 
def point_in_polygon(x, y, polygon):
    """Standard ray-casting point-in-polygon test. polygon is a list of
    (x, y) tuples, implicitly closed (last vertex connects to first)."""
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
        x1, y1 = x2, y2
    return inside
 
 
def _ray_boundary_distance(x, y, dx, dy, polygon, eps=1e-9):
    """Distance from (x, y) to the nearest polygon edge crossing along
    ray direction (dx, dy). Pass a unit vector for axis-aligned rays
    (as used here) so the returned distance is in mm. Returns None if
    no crossing is found (e.g. a self-intersecting/degenerate polygon,
    or the ray direction genuinely never exits the shape)."""
    best = None
    n = len(polygon)
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < eps:
            x1, y1 = x2, y2
            continue
        # Solve (x, y) + t*(dx, dy) = (x1, y1) + s*(ex, ey)
        t = ((x1 - x) * ey - (y1 - y) * ex) / denom
        s = ((x1 - x) * dy - (y1 - y) * dx) / denom
        if t > eps and -eps <= s <= 1 + eps:
            if best is None or t < best:
                best = t
        x1, y1 = x2, y2
    return best
 
 
def _axis_margin_ok(x, y, polygon, margin, axis):
    """Reproduces the rectangle/circle in_bounds_fn semantics: cast rays
    from (x, y) toward the boundary along this axis and require the
    NEARER of the two directions to clear the margin. This generalizes
    the circle's sqrt(R^2 - y^2) - |x| >= X_MARGIN check to an
    arbitrary polygon.
 
    Both directions are checked (not just the one the point is
    "leaning" toward) because for a concave boundary the true nearest
    edge on this axis can sit on either side of the point -- picking a
    single direction by the sign of x/y silently missed edges on the
    far side and could pass a point that was actually inside the
    margin. See RFC S6.4 / S8."""
    if axis == "x":
        d1 = _ray_boundary_distance(x, y, 1.0, 0.0, polygon)
        d2 = _ray_boundary_distance(x, y, -1.0, 0.0, polygon)
    else:
        d1 = _ray_boundary_distance(x, y, 0.0, 1.0, polygon)
        d2 = _ray_boundary_distance(x, y, 0.0, -1.0, polygon)
    dists = [d for d in (d1, d2) if d is not None]
    if not dists:
        return False
    return min(dists) >= margin - 1e-9
 
 
def make_in_bounds_fn(polygon, x_margin, y_margin):
    """Strict needle-tip test: matches generate_patch's in_bounds_fn contract."""
    def in_bounds(x, y):
        if not point_in_polygon(x, y, polygon):
            return False
        return (_axis_margin_ok(x, y, polygon, x_margin, "x")
                and _axis_margin_ok(x, y, polygon, y_margin, "y"))
    return in_bounds
 
 
def make_channel_bounds_fn(polygon, inset_polygon, inset_margin_mm):
    """Loose internal-branch-point test: matches generate_patch's
    channel_bounds_fn(x, y, margin) contract. If the requested margin
    matches the precomputed inset polygon's margin (the common case --
    generate_patch always calls with CHANNEL_EDGE_CLEARANCE), reuse it
    directly for a cheap point-in-polygon test. For any other margin,
    fall back to an on-the-fly axis-margin check against the raw
    polygon (slower, but still correct)."""
    def channel_bounds(x, y, margin):
        if inset_polygon is not None and inset_margin_mm is not None \
                and abs(margin - inset_margin_mm) < 1e-6:
            return point_in_polygon(x, y, inset_polygon)
        if not point_in_polygon(x, y, polygon):
            return False
        return (_axis_margin_ok(x, y, polygon, margin, "x")
                and _axis_margin_ok(x, y, polygon, margin, "y"))
    return channel_bounds
 
 
def bounding_half_extents(polygon):
    """Loose half_x/half_y reach, used only to size branch-tree depth
    (generate_patch's min_levels_for_reach) -- not a bounds gate."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    half_x = (max(xs) - min(xs)) / 2.0
    half_y = (max(ys) - min(ys)) / 2.0
    return half_x, half_y
 
 
def is_convex(polygon, eps=1e-9):
    """Cheap sanity check referenced in RFC S6.6: if the boundary is
    convex, the existing endpoint-only channel validity check
    (subtree_has_valid / build_channel) is safe as-is, same as for the
    rectangle/circle shapes. If not, a straight channel segment between
    two valid interior points can cut across a concave notch and exit
    the solid -- inspect (or mitigate per S6.6) before trusting output
    near concave regions."""
    n = len(polygon)
    if n < 4:
        return True
    sign = 0
    for i in range(n):
        x0, y0 = polygon[i - 1]
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) < eps:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True
 
 
if __name__ == "__main__":
    # Minimal smoke test against a synthetic square, runnable without
    # Fusion or any external dependency: `python wound_boundary.py`
    square = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    assert point_in_polygon(0, 0, square) is True
    assert point_in_polygon(10, 10, square) is False
    in_bounds = make_in_bounds_fn(square, x_margin=0.5, y_margin=1.0)
    assert in_bounds(0, 0) is True
    assert in_bounds(4.6, 0) is False   # within 0.5mm of the +x edge
    assert in_bounds(0, 4.1) is False   # within 1.0mm of the +y edge
    assert is_convex(square) is True
    concave = [(-5, -5), (5, -5), (5, 5), (0, 0), (-5, 5)]  # notch on top edge
    assert is_convex(concave) is False
 
    # Regression test: a point just below-and-right of the notch apex
    # used to pass the margin check even though the notch's near wall
    # sat only 0.45mm away (with a 0.5mm margin required), because the
    # old code only ever checked the +/-x or +/-y direction matching
    # the point's own sign, not whichever side the edge actually was on.
    notch_concave = [(-5, -5), (5, -5), (5, 5), (0, 0.2), (-5, 5)]
    in_bounds_notch = make_in_bounds_fn(notch_concave, x_margin=0.5, y_margin=0.5)
    assert in_bounds_notch(0.05, -0.25) is False
 
    print("wound_boundary.py smoke test passed.")