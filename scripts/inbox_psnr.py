# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hold-out PSNR restricted to the delivered region of interest.

Usage:
    python scripts/inbox_psnr.py <colmap_sparse_dir> <x0,y0,z0:x1,y1,z1> \
        <run_dir> [<run_dir> ...]

Why the full-image number misleads here
---------------------------------------
Once training is cropped to a region of interest, hold-out PSNR over the
whole frame stops measuring the thing being delivered. Measured on C:
the training crop box cut out-of-box splats 26% but cost 2.61 dB of
full-image PSNR and 0.064 SSIM (effect/noise 14.5 and 94.7) -- because the
evaluation renders the entire photograph, background included, and the box
deliberately starves everything outside it (Adam step x0.1, pixel loss
x0.1). Those background splats are not junk to the renderer; they are what
the training photographs actually contain. The operator removes them
because the DELIVERABLE is the subject, not the scene.

So the full-frame score and the cleanup score pull in opposite directions
by construction, and neither alone can say whether cropping is a good idea.
This measures the third thing: fidelity inside the region that actually
ships.

Method: project the eight corners of the crop box into each hold-out
camera, take the axis-aligned 2D hull of that projection clipped to the
frame, and score only those pixels. The 2D hull of a 3D box overestimates
the subject's silhouette, so this is conservative -- it still includes some
background, which biases the comparison TOWARD the uncropped arms.

The eval images are GT and render side by side (two panels of the camera's
own width, separated by 4px). Eval image i is images.txt entry 8i, which is
what --test-every 8 selects; verified by matching per-image dimensions.
"""

import math
import os
import sys

import numpy as np


def load_cameras(sparse_dir):
    """{camera_id: (w, h, fx, fy, cx, cy)}"""
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.txt"),
              encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            w, h = int(p[2]), int(p[3])
            fx, fy, cx, cy = (float(p[4]), float(p[5]), float(p[6]),
                              float(p[7]))
            cams[p[0]] = (w, h, fx, fy, cx, cy)
    return cams


def load_images(sparse_dir):
    """[(qvec, tvec, camera_id, name)] in file order."""
    out = []
    with open(os.path.join(sparse_dir, "images.txt"),
              encoding="utf-8", errors="replace") as fh:
        n = 0
        for line in fh:
            if line.startswith("#"):
                continue
            n += 1
            if n % 2 == 0:      # the POINTS2D line
                continue
            p = line.split()
            out.append((np.array([float(x) for x in p[1:5]]),
                        np.array([float(x) for x in p[5:8]]),
                        p[8], p[9]))
    return out


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def project_box(corners, q, t, fx, fy, cx, cy, w, h):
    """Axis-aligned pixel hull of the box, or None if fully behind/outside."""
    pts = (qvec2rotmat(q) @ corners.T).T + t
    front = pts[:, 2] > 1e-6
    if not front.any():
        return None
    pts = pts[front]
    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy
    x0, x1 = int(math.floor(u.min())), int(math.ceil(u.max()))
    y0, y1 = int(math.floor(v.min())), int(math.ceil(v.max()))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main(argv):
    from PIL import Image

    sparse, boxspec, runs = argv[0], argv[1], argv[2:]
    lo, hi = [[float(v) for v in half.split(",")]
              for half in boxspec.split(":")]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cams = load_cameras(sparse)
    imgs = load_images(sparse)
    print("box %s .. %s   cameras %d   images %d"
          % (lo, hi, len(cams), len(imgs)))

    for run in runs:
        name = os.path.basename(run.rstrip("\\/"))
        evdir = None
        for d in sorted(os.listdir(run)):
            if d.startswith("eval_step_"):
                evdir = os.path.join(run, d)
        if evdir is None:
            print("SKIP %s: no eval dir" % name, file=sys.stderr)
            continue
        full, inbox, skipped = [], [], 0
        for i in range(len(imgs) // 8 + 1):
            p = os.path.join(evdir, "%d.png" % i)
            if not os.path.isfile(p):
                continue
            q, t, cam_id, _nm = imgs[i * 8]
            w, h, fx, fy, cx, cy = cams[cam_id]
            im = np.asarray(Image.open(p).convert("RGB"))
            if im.shape[1] < 2 * w:
                skipped += 1
                continue
            gt = im[:, :w]
            rd = im[:, im.shape[1] - w:]
            full.append(psnr(gt, rd))
            box = project_box(corners, q, t, fx, fy, cx, cy, w, h)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            inbox.append(psnr(gt[y0:y1, x0:x1], rd[y0:y1, x0:x1]))
        print("%-16s full-frame %7.4f (n=%d)   IN-BOX %7.4f (n=%d)%s"
              % (name, np.mean(full), len(full),
                 np.mean(inbox), len(inbox),
                 "  skipped %d" % skipped if skipped else ""))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
