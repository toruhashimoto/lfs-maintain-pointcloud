# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score a trained model by how much hand cleanup it would still need.

Usage:
    python scripts/cleanup_cost.py <points3D.txt> <reference_clean.ply> \
        <run_dir_or_ply> [<run_dir_or_ply> ...]

Where the criterion comes from
------------------------------
The operator's own cleaned model is the label. Comparing
splat_111100.ply against splat_111100b.ply on the C capture, the manual
pass removed 76,530 of 5,000,000 splats (1.53%) and the removals split
cleanly in two:

  * 57,238 (74.8%) lay outside the bounding box of the kept model, and NOT
    ONE kept splat lay outside it. That boundary is exact.
  * 19,292 (25.2%) lay inside the box. Those are excluded from the score:
    the operator reports the in-box criterion varies per subject, so it is
    a judgement rather than a rule, and per-splat features cannot stand in
    for it anyway (max_axis > 0.01 catches 55% at 4% precision, distance to
    the surveyed cloud reaches only 16% precision).

So the score is the out-of-box count alone:

    out_of_box   visible splats outside the delivered model's box. Every one
                 is cleanup the operator had to do, and the boundary is
                 exact -- not one kept splat lay outside it.

Lower is better. Reported as a count and as a fraction of the visible model
so runs with different max_cap stay comparable. The in-box distance
percentiles are still printed, as a diagnostic of where the mass sits, but
they are NOT part of the score.

One thing the score cannot fix on its own: 29,175 of the 4,302,501 surveyed
points (0.68%) already lie outside the box. Rows anchored to those are held
OUTSIDE it, so position anchoring alone has a floor here -- cropping the
input cloud is what removes that floor.
"""

import json
import os
import sys

import numpy as np

_MULTIPLES = (2.0, 4.0, 8.0, 16.0)
_MIN_VISIBLE = 0.01     # the same gate the anchor pull uses


def read_ply(path):
    """(xyz, sigmoid(opacity), max_axis) for every vertex."""
    with open(path, "rb") as fh:
        blob = b""
        while b"end_header" not in blob:
            chunk = fh.read(4096)
            if not chunk:
                raise ValueError("no end_header in %s" % path)
            blob += chunk
    head = blob[:blob.index(b"end_header")]
    off = blob.index(b"end_header") + len(b"end_header")
    # Exactly one newline (optional CR before it). A while-loop here also
    # consumes the first payload byte whenever it happens to be 0x0A/0x0D --
    # the LSB of vertex 0's float32 x, so ~0.8% of real files.
    if blob[off:off + 1] == b"\r":
        off += 1
    if blob[off:off + 1] == b"\n":
        off += 1
    lines = head.split(b"\n")
    n = int([l.split()[-1] for l in lines
             if l.startswith(b"element vertex")][0])
    props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    if off + n * len(props) * 4 != os.path.getsize(path):
        raise ValueError("%s: unexpected size for %d x %d float32"
                         % (path, n, len(props)))
    m = np.memmap(path, dtype=np.float32, mode="r", offset=off,
                  shape=(n, len(props)))
    xyz = np.ascontiguousarray(m[:, 0:3])
    op = 1.0 / (1.0 + np.exp(-np.ascontiguousarray(
        m[:, props.index("opacity")]).astype(np.float64)))
    sc = np.ascontiguousarray(m[:, [props.index("scale_%d" % i)
                                    for i in (0, 1, 2)]])
    del m
    return xyz, op, np.exp(sc).max(axis=1)


def load_cloud(points3d_txt):
    cache = os.path.join(os.path.dirname(points3d_txt), "_cloud.npy")
    if os.path.isfile(cache):
        return np.load(cache)
    xs = []
    with open(points3d_txt, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split(None, 4)
            if len(f) >= 4:
                xs.append((float(f[1]), float(f[2]), float(f[3])))
    pts = np.asarray(xs, dtype=np.float32)
    np.save(cache, pts)
    return pts


def find_ply(arg):
    if os.path.isfile(arg):
        return arg
    best = None
    for name in sorted(os.listdir(arg)):
        if name.startswith("splat_") and name.endswith(".ply"):
            best = os.path.join(arg, name)
    return best


def main(argv):
    from scipy.spatial import cKDTree

    points3d, reference, targets = argv[0], argv[1], argv[2:]

    print("reference (delivered) model: %s" % os.path.basename(reference))
    ref_xyz, ref_op, _ = read_ply(reference)
    vis = ref_op >= _MIN_VISIBLE
    lo, hi = ref_xyz[vis].min(axis=0), ref_xyz[vis].max(axis=0)
    print("  box lo %s" % np.round(lo, 4))
    print("  box hi %s" % np.round(hi, 4))
    del ref_xyz, ref_op

    cloud = load_cloud(points3d)
    sub = cloud[::max(1, cloud.shape[0] // 200000)]
    spacing = float(np.median(
        cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]))
    print("  surveyed cloud %d points, median nn spacing %.6f"
          % (cloud.shape[0], spacing))
    tree = cKDTree(cloud)

    results = []
    for arg in targets:
        ply = find_ply(arg)
        if ply is None:
            print("SKIP %s: no splat ply" % arg, file=sys.stderr)
            continue
        name = os.path.basename(os.path.dirname(ply)) or os.path.basename(ply)
        xyz, op, maxax = read_ply(ply)
        v = op >= _MIN_VISIBLE
        nvis = int(v.sum())
        inbox = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        out_of_box = int((v & ~inbox).sum())

        sel = v & inbox
        dist, _ = tree.query(xyz[sel], workers=-1)
        rec = {"run": name, "ply": os.path.basename(ply),
               "rows": int(xyz.shape[0]), "visible": nvis,
               "spacing": spacing,
               "out_of_box": out_of_box,
               "out_of_box_frac": out_of_box / max(1, nvis)}
        for mult in _MULTIPLES:
            k = int(np.count_nonzero(dist > mult * spacing))
            rec["far_%gx" % mult] = k
            rec["far_%gx_frac" % mult] = k / max(1, nvis)
        for q, val in zip((50, 90, 95, 99),
                          np.percentile(dist, [50, 90, 95, 99])):
            rec["inbox_dist_p%d_x" % q] = float(val / spacing)
        results.append(rec)
        print("\n%s  (%s)" % (name, rec["ply"]))
        print("  visible %d / %d" % (nvis, rec["rows"]))
        print("  OUT OF BOX      %8d  (%.3f%% of visible)"
              % (out_of_box, 100 * rec["out_of_box_frac"]))
        print("  in-box dist to cloud (spacings): p50=%.2f p90=%.2f p95=%.2f p99=%.2f"
              % (rec["inbox_dist_p50_x"], rec["inbox_dist_p90_x"],
                 rec["inbox_dist_p95_x"], rec["inbox_dist_p99_x"]))
        for mult in _MULTIPLES:
            print("  in-box >%-4gx spacing %8d  (%.3f%%)   [diagnostic only]"
                  % (mult, rec["far_%gx" % mult],
                     100 * rec["far_%gx_frac" % mult]))
        del xyz, op, maxax

    out = os.path.join(os.path.dirname(os.path.abspath(targets[0])),
                       "cleanup_cost.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
