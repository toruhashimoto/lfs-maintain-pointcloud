# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Novel-view comparison: orbit path generation, rendering QA, quantification.

Why
---
Hold-out PSNR is measured on the capture ring, and every arm of every
campaign tied on it. Floaters and off-surface haze show up when the camera
leaves the ring -- which is exactly what a viewer user does. This renders
the SAME out-of-ring orbit through several trained models and puts numbers
on what appears outside the delivered region.

Pipeline (engine renders video; stills come from the bundled FFmpeg DLLs):

    python scripts/novelview.py path --out orbit.json \
        --center 0.583,-0.548,0 --radius 2.1,2.3 --heights -1.35,-0.20
    LichtFeld-Studio.exe --render-camera-path orbit.json \
        --render-load splat_30000.ply --render-output out.mp4 \
        --render-width 1280 --render-height 720 --render-fps 24
    mp4frames out.mp4 framedir 12          # scratch tool, see repo docs
    python scripts/novelview.py count --frames framedir --path orbit.json \
        --box "-0.9992,-1.4594,-1.0:2.165,0.3634,0.9999"
    python scripts/novelview.py collage --dirs a,b,c --labels A,B,C \
        --indices 0,8,16 --out cmp.png

Geometry facts this encodes (verified against the engine source):
  * keyframe rotation is CAMERA-TO-WORLD, quaternion stored as [w,x,y,z]
    (timeline.cpp saveToJson), position is the camera centre in world.
  * the render camera is COLMAP-convention: +Z forward, +Y down. World up
    is -Y for these captures (RealityScan alignment).
  * focal_length_mm is 35mm-equivalent with SENSOR_HEIGHT_35MM = 24.0:
    fov_y = 2*atan(12/f). fx is derived from fov_x via the aspect ratio
    (coordinate_conventions.hpp computePixelFocalLengths).
  * frames are sampled at t = frame/fps and the spline passes through the
    keyframes, so with keyframes every K/fps seconds, every K-th frame sits
    EXACTLY on a keyframe. count only looks at those frames, which is what
    makes the pose recoverable without reimplementing the C++ spline.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

KEYFRAME_STEP_S = 0.5     # keyframe every 0.5 s
FPS = 24                  # render fps; every 12th frame is exactly a keyframe
FRAMES_PER_KF = int(round(KEYFRAME_STEP_S * FPS))


# ------------------------------------------------------------------ geometry

def look_at_c2w(eye, target):
    """Camera-to-world rotation (3x3) for a y-down world, COLMAP camera."""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f /= np.linalg.norm(f)
    d0 = np.array([0.0, 1.0, 0.0])            # world +y = physically down
    d = d0 - f * float(d0 @ f)
    n = np.linalg.norm(d)
    if n < 1e-9:                              # looking straight down/up
        d0 = np.array([1.0, 0.0, 0.0])
        d = d0 - f * float(d0 @ f)
        n = np.linalg.norm(d)
    d /= n
    r = np.cross(d, f)                        # right = down x forward
    return np.stack([r, d, f], axis=1)        # columns: right, down, forward


def mat_to_quat_wxyz(m):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return [float(w), float(x), float(y), float(z)]


def pixel_focals(width, height, focal_mm):
    fov_y = 2.0 * math.atan(24.0 / (2.0 * focal_mm))
    aspect = width / height
    fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * aspect)
    fx = width / (2.0 * math.tan(fov_x * 0.5))
    fy = height / (2.0 * math.tan(fov_y * 0.5))
    return fx, fy


# ---------------------------------------------------------------------- path

def cmd_path(args):
    cx, cy, cz = (float(v) for v in args.center.split(","))
    radii = [float(v) for v in args.radius.split(",")]
    heights = [float(v) for v in args.heights.split(",")]
    if not radii or len(radii) != len(heights):
        raise SystemExit("--radius and --heights need the same nonzero count")
    if args.keyframes < 1:
        raise SystemExit("--keyframes must be >= 1")
    per_lap = args.keyframes

    keyframes = []
    t = 0.0
    for radius, y in zip(radii, heights):
        for i in range(per_lap):
            ang = 2.0 * math.pi * i / per_lap
            eye = [cx + radius * math.cos(ang), y,
                   cz + radius * math.sin(ang)]
            quat = mat_to_quat_wxyz(look_at_c2w(eye, [cx, cy, cz]))
            keyframes.append({
                "time": round(t, 6),
                "position": [round(v, 6) for v in eye],
                "rotation": [round(v, 9) for v in quat],
                "focal_length_mm": args.focal,
                "easing": 0,
            })
            t += KEYFRAME_STEP_S
    doc = {"version": 4,
           "clip_duration": keyframes[-1]["time"],
           "keyframes": keyframes}
    # LF only. The engine's timeline loader sizes its read from the on-disk
    # byte count and then reads in text mode; CRLF shrinks what it gets back
    # and it refuses the file as "changed size while it was being read".
    with open(args.out, "w", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
    # The engine schedules ceil(duration*fps)+1 frames but the decoder gets
    # one fewer back, so the LAST keyframe never yields a measured view
    # (63 of 64 in practice). Harmless for A/B -- identical for every arm.
    print("wrote %s: %d keyframes, %.1f s, ~%d frames at %d fps "
          "(every %dth frame is a keyframe; the last keyframe has no frame)"
          % (args.out, len(keyframes), doc["clip_duration"],
             int(math.ceil(doc["clip_duration"] * FPS)) + 1, FPS,
             FRAMES_PER_KF))
    return 0


# --------------------------------------------------------------------- count

def _load_keyframes(path_json):
    with open(path_json) as fh:
        doc = json.load(fh)
    return doc["keyframes"]


def _hull_of_box(kf, box, width, height, focal_mm):
    """Axis-aligned pixel hull of the box in this keyframe's camera.

    Assumes every box corner is in front of the camera; corners behind the
    near plane are dropped rather than clipped, which would under-cover the
    silhouette and inflate the out-of-box count. Holds for the orbits used
    here (camera 2.1-2.6 m from the box centre, corners ~1.9 m out); a
    tighter orbit that enters the box needs edge clipping first.
    """
    lo, hi = box
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w, x, y, z = kf["rotation"]
    from scipy.spatial.transform import Rotation
    r_c2w = Rotation.from_quat([x, y, z, w]).as_matrix()
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ np.asarray(kf["position"], dtype=np.float64)
    pts = corners @ r_w2c.T + t_w2c
    front = pts[:, 2] > 1e-6
    if not front.any():
        return None
    pts = pts[front]
    fx, fy = pixel_focals(width, height, focal_mm)
    u = fx * pts[:, 0] / pts[:, 2] + width * 0.5
    v = fy * pts[:, 1] / pts[:, 2] + height * 0.5
    x0 = max(0, int(math.floor(u.min())))
    x1 = min(width, int(math.ceil(u.max())))
    y0 = max(0, int(math.floor(v.min())))
    y1 = min(height, int(math.ceil(v.max())))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def cmd_count(args):
    from PIL import Image
    from scipy import ndimage

    box = [[float(v) for v in half.split(",")]
           for half in args.box.split(":")]
    kfs = _load_keyframes(args.path)
    rows = []
    for name in sorted(os.listdir(args.frames)):
        if not name.startswith("frame_") or not name.endswith(".ppm"):
            continue
        fidx = int(name[6:11])
        if fidx % FRAMES_PER_KF:
            continue                      # only frames that sit on keyframes
        kidx = fidx // FRAMES_PER_KF
        if kidx >= len(kfs):
            continue
        im = np.asarray(Image.open(os.path.join(args.frames, name)))
        h, w = im.shape[:2]
        # The focal comes from the keyframe itself: the JSON that drove the
        # render already stores it, and a --focal that disagreed with the
        # path would silently misplace every hull.
        focal = kfs[kidx].get("focal_length_mm", args.focal)
        hull = _hull_of_box(kfs[kidx], box, w, h, focal)
        lit = im.max(axis=2) > args.threshold
        outside = lit.copy()
        if hull is not None:
            x0, y0, x1, y1 = hull
            outside[y0:y1, x0:x1] = False
        # isolated specks: small connected components fully outside the hull
        lab, nlab = ndimage.label(lit)
        specks = 0
        if nlab:
            areas = ndimage.sum_labels(np.ones_like(lab), lab,
                                       index=np.arange(1, nlab + 1))
            inside_any = ndimage.maximum(~outside & lit, lab,
                                         index=np.arange(1, nlab + 1))
            specks = int(np.sum((areas < args.speck_area)
                                & (inside_any == 0)))
        rows.append((kidx, int(outside.sum()), specks,
                     int(lit.sum())))
    if not rows:
        print("no keyframe-aligned frames found in %s" % args.frames)
        return 1
    rows.sort()
    out_px = np.array([r[1] for r in rows], dtype=np.float64)
    specks = np.array([r[2] for r in rows], dtype=np.float64)
    print("%s: %d keyframe views" % (args.frames, len(rows)))
    print("  out-of-box lit pixels  mean %9.0f  median %9.0f  p90 %9.0f"
          % (out_px.mean(), np.median(out_px), np.percentile(out_px, 90)))
    print("  isolated specks (<%dpx) mean %8.1f  median %8.1f  max %5.0f"
          % (args.speck_area, specks.mean(), np.median(specks),
             specks.max()))
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("keyframe,out_px,specks,lit_px\n")
            for r in rows:
                fh.write("%d,%d,%d,%d\n" % r)
        print("  wrote %s" % args.csv)
    return 0


# ------------------------------------------------------------------- collage

def cmd_collage(args):
    from PIL import Image, ImageDraw

    dirs = args.dirs.split(",")
    labels = args.labels.split(",")
    indices = [int(v) for v in args.indices.split(",")]
    if len(dirs) != len(labels):
        raise SystemExit("--dirs and --labels need the same count")
    rows = []
    for d, label in zip(dirs, labels):
        cells = []
        for kidx in indices:
            p = os.path.join(d, "frame_%05d.ppm" % (kidx * FRAMES_PER_KF))
            im = Image.open(p).convert("RGB")
            if args.scale != 1.0:
                im = im.resize((int(im.width * args.scale),
                                int(im.height * args.scale)))
            cells.append(im)
        strip = Image.new("RGB", (sum(c.width for c in cells) +
                                  4 * (len(cells) - 1),
                                  cells[0].height + 22), (24, 24, 24))
        x = 0
        for c in cells:
            strip.paste(c, (x, 22))
            x += c.width + 4
        ImageDraw.Draw(strip).text((6, 4), label, fill=(255, 255, 255))
        rows.append(strip)
    total = Image.new("RGB", (max(r.width for r in rows),
                              sum(r.height for r in rows) +
                              4 * (len(rows) - 1)), (24, 24, 24))
    y = 0
    for r in rows:
        total.paste(r, (0, y))
        y += r.height + 4
    total.save(args.out)
    print("wrote %s (%dx%d)" % (args.out, total.width, total.height))
    return 0


def _glue_negative_values(argv):
    """Rewrite ``--flag -1.35,...`` as ``--flag=-1.35,...``.

    Same argparse limitation crop_input.py hit: a value starting with ``-``
    is classified as another option. Heights, centers, and boxes all start
    with minus signs in this scene's frame.
    """
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if (a.startswith("--") and "=" not in a and i + 1 < len(argv)
                and argv[i + 1][:1] == "-" and len(argv[i + 1]) > 1
                and (argv[i + 1][1].isdigit() or argv[i + 1][1] == ".")):
            out.append(a + "=" + argv[i + 1])
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("path", help="generate an orbit camera-path JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--center", required=True, help="cx,cy,cz (orbit target)")
    p.add_argument("--radius", required=True, help="one radius per lap, comma-sep")
    p.add_argument("--heights", required=True, help="one camera y per lap, comma-sep")
    p.add_argument("--keyframes", type=int, default=32, help="keyframes per lap")
    p.add_argument("--focal", type=float, default=21.0, help="35mm-equiv focal")
    p.set_defaults(fn=cmd_path)

    p = sub.add_parser("count", help="quantify out-of-box rendered matter")
    p.add_argument("--frames", required=True, help="dir of frame_XXXXX.ppm")
    p.add_argument("--path", required=True, help="the orbit JSON used to render")
    p.add_argument("--box", required=True, help="x0,y0,z0:x1,y1,z1 delivered box")
    p.add_argument("--focal", type=float, default=21.0,
                   help="fallback only; the keyframe's own focal_length_mm "
                        "wins when present")
    p.add_argument("--threshold", type=int, default=12,
                   help="max-RGB above this = rendered matter (h264 noise floor)")
    p.add_argument("--speck-area", type=int, default=400)
    p.add_argument("--csv", default=None)
    p.set_defaults(fn=cmd_count)

    p = sub.add_parser("collage", help="side-by-side comparison sheet")
    p.add_argument("--dirs", required=True, help="comma-sep frame dirs")
    p.add_argument("--labels", required=True, help="comma-sep row labels")
    p.add_argument("--indices", required=True, help="comma-sep KEYFRAME indices")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_collage)

    args = ap.parse_args(_glue_negative_values(
        list(sys.argv[1:] if argv is None else argv)))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
