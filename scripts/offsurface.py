# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""How far the splats the plugin never touches sit from the surveyed surface.

Usage:
    python scripts/offsurface.py <points3D.txt> <run_dir> [<run_dir> ...]

Why this measurement
--------------------
Everything measured so far shows the leash holds the INITIAL rows near their
anchors, and that it costs nothing photometrically. Neither says the model
is better: hold-out PSNR is identical between arms, so at evaluation views
the two are indistinguishable. The claim that actually matters for a
measurement-grade model -- "there is no geometry here that is not on the
real object" -- has never been tested.

Testing it with the initial rows would be circular: the leash holds them
near their anchors, and their anchors ARE cloud points, so of course they
sit on the surface. The rows MRNF appends are different. With
`anchor_new_splats = False` (the default) they are never anchored, never
pulled, and never measured by init_drift. At --max-cap 16000000 they are
8.1M of the 16M rows -- half the model, untouched by the plugin.

So: how far is each appended splat from the nearest point of the surveyed
cloud? If anchoring the initial rows also keeps the grown population on the
surface, that distribution tightens. If it does not, the plugin's effect is
confined to the rows it holds directly -- which is equally worth knowing.

This is the opposite direction to the coverage metric README rejects. That
one is cloud-point -> nearest splat, which merely re-measures that splats
sit on their own anchors. This is splat -> nearest cloud point, over splats
that have no anchor.

Distances are reported in multiples of the cloud's own median
nearest-neighbour spacing, so they read the same way as `escaped_*`.
"""

import json
import os
import sys

import numpy as np

# 62 float32 properties per vertex: xyz, normals, f_dc[3], f_rest[45],
# opacity, scale[3], rot[4]. Confirmed against the emitted header.
_STRIDE = 62
_OPACITY_COL = 54
_MULTIPLES = (1.0, 2.0, 4.0, 8.0)
_QUANTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)


def load_initial_cloud(points3d_txt, cache):
    """[N,3] float32 of the COLMAP points, cached as .npy after first parse."""
    if os.path.isfile(cache):
        return np.load(cache)
    xs = []
    with open(points3d_txt, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split(None, 4)
            if len(f) < 4:
                continue
            xs.append((float(f[1]), float(f[2]), float(f[3])))
    pts = np.asarray(xs, dtype=np.float32)
    np.save(cache, pts)
    return pts


def ply_columns(path):
    """(xyz, opacity_sigmoid) for every vertex, memory-mapped."""
    with open(path, "rb") as fh:
        blob = b""
        while b"end_header" not in blob:
            chunk = fh.read(4096)
            if not chunk:
                raise ValueError("no end_header in %s" % path)
            blob += chunk
    offset = blob.index(b"end_header") + len(b"end_header\n")
    n = int([l.split()[-1] for l in blob.split(b"\n")
             if l.startswith(b"element vertex")][0])
    expected = offset + n * _STRIDE * 4
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError("%s: size %d != expected %d (stride assumption wrong)"
                         % (path, actual, expected))
    m = np.memmap(path, dtype=np.float32, mode="r", offset=offset,
                  shape=(n, _STRIDE))
    xyz = np.array(m[:, 0:3])
    op = 1.0 / (1.0 + np.exp(-np.array(m[:, _OPACITY_COL], dtype=np.float64)))
    del m
    return xyz, op


def summarise(dist, spacing, label):
    out = {"label": label, "rows": int(dist.size)}
    if dist.size == 0:
        return out
    q = np.percentile(dist, _QUANTILES)
    for name, v in zip(("p50", "p75", "p90", "p95", "p99", "p999"), q):
        out[name] = float(v)
        out[name + "_x"] = float(v / spacing) if spacing else 0.0
    for mult in _MULTIPLES:
        n = int(np.count_nonzero(dist > mult * spacing))
        out["beyond_%gx" % mult] = n
        out["beyond_%gx_frac" % mult] = float(n) / dist.size
    return out


def main(argv):
    from scipy.spatial import cKDTree

    points3d, run_dirs = argv[0], argv[1:]
    cache = os.path.join(os.path.dirname(points3d), "_initial_cloud.npy")
    print("loading initial cloud ...", flush=True)
    cloud = load_initial_cloud(points3d, cache)
    n0 = cloud.shape[0]
    print("  %d points" % n0, flush=True)

    # Same statistic the plugin derives its dead zone from, recomputed here
    # so the two are on one scale.
    sub = cloud[::max(1, n0 // 200000)]
    spacing = float(np.median(cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]))
    print("  median nn spacing %.6f" % spacing, flush=True)

    print("building KD-tree over the surveyed cloud ...", flush=True)
    tree = cKDTree(cloud)

    results = []
    for d in run_dirs:
        ply = None
        for name in sorted(os.listdir(d)):
            if name.startswith("splat_") and name.endswith(".ply"):
                ply = os.path.join(d, name)
        if ply is None:
            print("SKIP %s: no splat ply" % d, file=sys.stderr)
            continue
        run = os.path.basename(d.rstrip("\\/"))
        print("\n%s  (%s)" % (run, os.path.basename(ply)), flush=True)
        xyz, op = ply_columns(ply)
        print("  %d rows, %d appended" % (xyz.shape[0], max(0, xyz.shape[0] - n0)),
              flush=True)

        rec = {"run": run, "rows": int(xyz.shape[0]),
               "spacing": spacing, "arms": []}
        for label, sl in (("appended", slice(n0, None)),
                          ("initial", slice(0, n0))):
            p, o = xyz[sl], op[sl]
            if p.shape[0] == 0:
                continue
            vis = o >= 0.01          # same gate the pull uses
            for vlabel, mask in (("all", np.ones(p.shape[0], bool)),
                                 ("visible", vis)):
                if not mask.any():
                    continue
                dist, _ = tree.query(p[mask], workers=-1)
                s = summarise(dist, spacing, "%s/%s" % (label, vlabel))
                rec["arms"].append(s)
                print("    %-18s n=%9d  p50=%.5f (%.2fx)  p95=%.5f (%.2fx)  "
                      ">4x: %d (%.3f%%)"
                      % (s["label"], s["rows"], s["p50"], s["p50_x"],
                         s["p95"], s["p95_x"], s["beyond_4x"],
                         100 * s["beyond_4x_frac"]), flush=True)
        results.append(rec)
        del xyz, op

    out = os.path.join(os.path.dirname(os.path.abspath(run_dirs[0])),
                       "offsurface.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
