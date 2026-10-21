import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
import cv2
 
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None
 
 
def load_mask_from_image(image_path):
    """Very simple Otsu-threshold segmentation. Swap this out for a
    proper model if you have one -- everything downstream only cares
    that this returns a single-channel uint8 mask where wound pixels
    are 255 and everything else is 0."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask
 
 
def load_mask_from_file(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask
 
 
def extract_boundary_polygon_px(mask, approx_epsilon_px=2.0):
    """Largest external contour -> simplified polygon, in pixel coords."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask -- check segmentation input.")
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, approx_epsilon_px, closed=True)
    pts = approx.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        raise ValueError("Segmented boundary has fewer than 3 vertices after simplification -- "
                          "try a smaller --approx-epsilon-px.")
    return pts  # Nx2, pixel coords, (col, row)
 
 
def px_line_length(p1, p2):
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
 
 
def polygon_px_to_mm(pts_px, mm_per_px_x, mm_per_px_y):
    pts_mm = pts_px.copy()
    pts_mm[:, 0] *= mm_per_px_x
    # Image row axis grows downward; flip so the patch's +y is "up",
    # matching the coordinate convention the rest of the toolchain
    # (and the original rectangle/circle generator) uses.
    pts_mm[:, 1] *= -mm_per_px_y
    return pts_mm
 
 
def center_on_centroid(pts_mm):
    centroid = pts_mm.mean(axis=0)
    return pts_mm - centroid, centroid
 
 
def compute_inset_polygon(pts_mm, margin_mm):
    """Inward offset of the boundary by margin_mm, used for channel
    branch-point clearance. Requires shapely -- polygon offsetting for
    concave shapes is not something worth hand-rolling here."""
    if Polygon is None:
        print("WARNING: shapely not installed -- skipping inset polygon. "
              "channel_bounds_fn will fall back to an on-the-fly check "
              "against the raw boundary (slower, but still correct).",
              file=sys.stderr)
        return None
    poly = Polygon(pts_mm)
    if not poly.is_valid:
        poly = poly.buffer(0)  # common fix for minor self-intersections
    inset = poly.buffer(-margin_mm, join_style=2)  # mitre join
    if inset.is_empty:
        print("WARNING: inset polygon is empty at this margin -- the "
              "segmented area may be too small/narrow for the requested "
              "channel clearance.", file=sys.stderr)
        return None
    if inset.geom_type == "MultiPolygon":
        inset = max(inset.geoms, key=lambda g: g.area)
    return np.array(inset.exterior.coords[:-1])  # drop repeated closing vertex
 
 
def build_boundary_dict(mask, mm_per_px_x, mm_per_px_y, channel_clearance_mm,
                         approx_epsilon_px=2.0, source_meta=None):
    """Core mask -> boundary.json-shaped-dict conversion, factored out of
    main() so other scripts (e.g. a batch pipeline driven by CVAT
    annotations) can call this directly per-mask instead of shelling out
    to this file's CLI once per image. Returns the same dict shape main()
    writes to boundary.json, plus (pts_mm, inset_mm) for anyone who also
    wants to render a preview without re-deriving them."""
    pts_px = extract_boundary_polygon_px(mask, approx_epsilon_px)
    pts_mm = polygon_px_to_mm(pts_px, mm_per_px_x, mm_per_px_y)
    pts_mm, centroid_mm = center_on_centroid(pts_mm)
    inset_mm = compute_inset_polygon(pts_mm, channel_clearance_mm)
 
    boundary = {
        "units": "mm",
        "polygon": [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in pts_mm],
        "inset_polygon": (
            [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in inset_mm]
            if inset_mm is not None else None
        ),
        "inset_margin_mm": channel_clearance_mm,
        "source": {
            "mm_per_px_x": mm_per_px_x,
            "mm_per_px_y": mm_per_px_y,
            "vertex_count": len(pts_mm),
            **(source_meta or {}),
        },
    }
    return boundary, pts_mm, inset_mm
 
 
def frame_calibration(mask, frame_width_mm, frame_height_mm):
    """Method (a) calibration: treat the whole mask's pixel dimensions
    as a known, fixed physical frame size. Returns (mm_per_px_x, mm_per_px_y)."""
    height_px, width_px = mask.shape[:2]
    return frame_width_mm / width_px, frame_height_mm / height_px
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single wound photo to segment with a basic threshold.")
    src.add_argument("--mask", help="Single pre-made binary mask (wound = white).")
    src.add_argument("--mask-dir", help="Folder of pre-made binary masks to batch-process "
                                         "(fixed-frame calibration only -- see --frame-width-mm).")
 
    cal = ap.add_argument_group("calibration (pick exactly one method)")
    cal.add_argument("--frame-width-mm", type=float, default=None,
                      help="Method (a): real-world width of the whole capture frame, in mm. "
                           "Requires --frame-height-mm too.")
    cal.add_argument("--frame-height-mm", type=float, default=None,
                      help="Method (a): real-world height of the whole capture frame, in mm.")
    cal.add_argument("--ref-line-px", nargs=4, type=float, default=None,
                      metavar=("X1", "Y1", "X2", "Y2"),
                      help="Method (b): pixel coords of a reference object of known length.")
    cal.add_argument("--ref-line-mm", type=float, default=None,
                      help="Method (b): real-world length in mm of the reference line.")
    cal.add_argument("--ref-line-px-y", nargs=4, type=float, default=None,
                      metavar=("X1", "Y1", "X2", "Y2"),
                      help="Method (b), optional: separate reference line for y-axis scale.")
    cal.add_argument("--ref-line-mm-y", type=float, default=None,
                      help="Method (b), optional: real-world length in mm of the y-axis reference line.")
 
    ap.add_argument("--channel-clearance-mm", type=float, default=0.215,
                     help="Inward margin for the channel-branch inset polygon "
                          "(default matches CHANNEL_EDGE_CLEARANCE in the generator: "
                          "channel_hub_radius(0.1) * OVERLAP_FACTOR(1.15) + CHANNEL_WALL_SAFETY(0.1)).")
    ap.add_argument("--approx-epsilon-px", type=float, default=2.0,
                     help="cv2.approxPolyDP epsilon, in pixels, for contour simplification.")
    ap.add_argument("--out", default="boundary.json", help="Output path (single-file mode only).")
    ap.add_argument("--out-dir", default=None, help="Output folder (batch mode only; "
                                                      "default: <mask-dir>/boundaries).")
    ap.add_argument("--preview", nargs="?", const=True, default=None,
                     help="Save a PNG preview. Single-file mode: give a path, or omit the "
                          "value to use <out>_preview.png. Batch mode: pass with no value to "
                          "save one preview per mask alongside its boundary_*.json.")
    args = ap.parse_args()
 
    using_frame = args.frame_width_mm is not None or args.frame_height_mm is not None
    using_refline = args.ref_line_px is not None or args.ref_line_mm is not None
    if using_frame and using_refline:
        ap.error("Use either --frame-width-mm/--frame-height-mm OR --ref-line-px/--ref-line-mm, not both.")
    if using_frame and (args.frame_width_mm is None or args.frame_height_mm is None):
        ap.error("--frame-width-mm and --frame-height-mm must be given together.")
    if using_refline and (args.ref_line_px is None or args.ref_line_mm is None):
        ap.error("--ref-line-px and --ref-line-mm must be given together.")
    if not using_frame and not using_refline:
        ap.error("No calibration given -- use --frame-width-mm/--frame-height-mm or --ref-line-px/--ref-line-mm.")
    if args.mask_dir and using_refline:
        ap.error("--mask-dir (batch mode) only supports --frame-width-mm/--frame-height-mm calibration, "
                  "since --ref-line-px would need a different pixel lookup per image.")
 
    def calibrate(mask):
        if using_frame:
            return frame_calibration(mask, args.frame_width_mm, args.frame_height_mm)
        x1, y1, x2, y2 = args.ref_line_px
        mm_per_px_x = args.ref_line_mm / px_line_length((x1, y1), (x2, y2))
        if args.ref_line_px_y and args.ref_line_mm_y:
            xy1 = args.ref_line_px_y[0:2]
            xy2 = args.ref_line_px_y[2:4]
            mm_per_px_y = args.ref_line_mm_y / px_line_length(xy1, xy2)
        else:
            mm_per_px_y = mm_per_px_x
        return mm_per_px_x, mm_per_px_y
 
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        out_dir = Path(args.out_dir) if args.out_dir else mask_dir / "boundaries"
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_paths = sorted(mask_dir.glob("*.png"))
        if not mask_paths:
            print(f"No .png files found in {mask_dir}", file=sys.stderr)
            return
        generated, errors = 0, []
        for mask_path in mask_paths:
            try:
                mask = load_mask_from_file(str(mask_path))
                mm_per_px_x, mm_per_px_y = calibrate(mask)
                boundary, pts_mm, inset_mm = build_boundary_dict(
                    mask, mm_per_px_x, mm_per_px_y, args.channel_clearance_mm,
                    approx_epsilon_px=args.approx_epsilon_px,
                    source_meta={"mask": str(mask_path)},
                )
                stem = mask_path.stem
                boundary_path = out_dir / f"boundary_{stem}.json"
                with open(boundary_path, "w") as f:
                    json.dump(boundary, f, indent=2)
                if args.preview:
                    save_preview(pts_mm, inset_mm, str(out_dir / f"boundary_{stem}_preview.png"))
                generated += 1
                print(f"Generated: {boundary_path.name} ({len(pts_mm)} vertices, "
                      f"{mm_per_px_x:.4f}x{mm_per_px_y:.4f} mm/px)")
            except Exception as exc:
                errors.append(f"{mask_path.name}: {exc}")
        print()
        print(f"Boundaries generated: {generated}/{len(mask_paths)}")
        if errors:
            print("Errors:")
            for message in errors:
                print(f"  - {message}")
        return
 
    mask = load_mask_from_image(args.image) if args.image else load_mask_from_file(args.mask)
    mm_per_px_x, mm_per_px_y = calibrate(mask)
 
    boundary, pts_mm, inset_mm = build_boundary_dict(
        mask, mm_per_px_x, mm_per_px_y, args.channel_clearance_mm,
        approx_epsilon_px=args.approx_epsilon_px,
        source_meta={"image": args.image or args.mask},
    )
 
    with open(args.out, "w") as f:
        json.dump(boundary, f, indent=2)
    print(f"Wrote {args.out}: {len(pts_mm)} vertices, "
          f"scale {mm_per_px_x:.4f} mm/px (x), {mm_per_px_y:.4f} mm/px (y)")
 
    if args.preview:
        preview_path = args.preview if isinstance(args.preview, str) else str(Path(args.out).with_suffix("")) + "_preview.png"
        save_preview(pts_mm, inset_mm, preview_path)
 
 
def save_preview(pts_mm, inset_mm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(6, 6))
    poly_closed = np.vstack([pts_mm, pts_mm[0]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], "b-o", markersize=3, label="wound boundary")
    if inset_mm is not None:
        inset_closed = np.vstack([inset_mm, inset_mm[0]])
        ax.plot(inset_closed[:, 0], inset_closed[:, 1], "r--", label="channel inset")
    ax.scatter([0], [0], c="k", marker="+", s=80, label="origin (centroid)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend()
    ax.set_title(f"Extracted wound boundary ({len(pts_mm)} vertices)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Preview saved to {out_path}")
 
 
if __name__ == "__main__":
    main()
 