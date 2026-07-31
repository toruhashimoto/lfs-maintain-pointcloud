# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Split the operator's manual removals into out-of-box and in-box.

Usage:
    python scripts/removed_split.py <raw.ply> <cleaned.ply> [<points3D.txt>]

What it establishes, and what it does not
-----------------------------------------
The cleaned PLY is the raw one with rows deleted, so raw and cleaned rows
match on their exact float32 XYZ and the removed set is recoverable.

The useful number is **what fraction of the removals a box explains**: 74.8%
on C, 80.3% on D. That is what justifies cropping the input cloud to
the box, because everything outside it is cleanup the operator had to do
anyway.

Note what is NOT evidence. "Not one kept splat lies outside the box" is true
by construction -- the box IS the bounding box of the kept visible splats --
and so is the equality between "visible raw splats outside the box" and
"visible removals outside the box". Both are printed as consistency checks
against cleanup_cost.py, not as findings.

The in-box base rate is the number that decides whether per-splat features
could ever stand in for the in-box judgement. On C it is 0.39%, and at
that rate max_axis reached 4% precision and distance-to-cloud 16%. On D
it is 1.823%, 4.7x higher, which is the better place to retry.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleanup_cost import read_ply  # noqa: E402

MIN_VISIBLE = 0.01              # the same gate the anchor pull uses
DT = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])

# Confirmed against both datasets' delivered boxes. Centre [1,-0.5,0],
# size [4,2,2]. Reported here so a new capture can be checked against it.
NOMINAL = (np.array([-1.0, -1.5, -1.0]), np.array([3.0, 0.5, 1.0]))


def _keys(xyz):
    return np.ascontiguousarray(xyz, dtype=np.float32).view(DT).ravel()


def main(argv):
    raw_path, clean_path = argv[0], argv[1]
    points3d = argv[2] if len(argv) > 2 else None

    raw_xyz, raw_op, _ = read_ply(raw_path)
    cl_xyz, cl_op, _ = read_ply(clean_path)
    print("raw     %d splats  (%s)" % (len(raw_xyz), os.path.basename(raw_path)))
    print("cleaned %d splats  (%s)" % (len(cl_xyz), os.path.basename(clean_path)))

    kept = np.isin(_keys(raw_xyz), _keys(cl_xyz))
    n_removed = int((~kept).sum())
    print("removed %d  (%.2f%% of raw)"
          % (n_removed, 100.0 * n_removed / len(raw_xyz)))
    if int(kept.sum()) != len(cl_xyz):
        print("  NOTE %+d vs the cleaned count -- duplicate positions exist"
              % (int(kept.sum()) - len(cl_xyz)))

    vis_cl = cl_op >= MIN_VISIBLE
    lo, hi = cl_xyz[vis_cl].min(axis=0), cl_xyz[vis_cl].max(axis=0)
    print("\ndelivered box (bbox of visible kept splats)")
    print("  lo     %9.4f %9.4f %9.4f" % tuple(lo))
    print("  hi     %9.4f %9.4f %9.4f" % tuple(hi))
    print("  size   %9.4f %9.4f %9.4f" % tuple(hi - lo))
    print("  centre %9.4f %9.4f %9.4f" % tuple((hi + lo) / 2))

    def inbox(xyz):
        return np.all((xyz >= lo) & (xyz <= hi), axis=1)

    rm_in = inbox(raw_xyz[~kept])
    print("\nremovals explained by the box")
    print("  outside %8d  (%.1f%%)   <- what cropping the input attacks"
          % ((~rm_in).sum(), 100.0 * (~rm_in).sum() / n_removed))
    print("  inside  %8d  (%.1f%%)   <- per-subject judgement"
          % (rm_in.sum(), 100.0 * rm_in.sum() / n_removed))

    vis_raw = raw_op >= MIN_VISIBLE
    raw_in = inbox(raw_xyz)
    denom = int((vis_raw & raw_in).sum())
    num = int((vis_raw & raw_in & ~kept).sum())
    print("\nin-box removal base rate  %d / %d = %.3f%%"
          % (num, denom, 100.0 * num / max(denom, 1)))
    print("consistency (both tautological, must match cleanup_cost.py)")
    print("  visible raw splats outside the box %8d"
          % int((vis_raw & ~raw_in).sum()))
    print("  visible kept splats outside the box %7d  (must be 0)"
          % int((~inbox(cl_xyz[vis_cl])).sum()))

    print("\nnominal box %s : %s"
          % (",".join("%g" % v for v in NOMINAL[0]),
             ",".join("%g" % v for v in NOMINAL[1])))
    print("  contains the delivered box: %s"
          % bool(np.all(NOMINAL[0] <= lo) and np.all(NOMINAL[1] >= hi)))
    print("  lo gap %s" % np.round(lo - NOMINAL[0], 4))
    print("  hi gap %s" % np.round(NOMINAL[1] - hi, 4))

    if points3d:
        from cleanup_cost import load_cloud
        cloud = load_cloud(points3d)
        print("\nsurveyed points outside a box (what crop_input.py drops)")
        for label, (blo, bhi) in (("delivered", (lo, hi)), ("nominal", NOMINAL)):
            for pad in (0.0, 0.05):
                ext = (bhi - blo) * pad
                plo, phi = blo - ext, bhi + ext
                out = int((~np.all((cloud >= plo) & (cloud <= phi),
                                   axis=1)).sum())
                print("  %-9s pad %3.0f%%  %8d of %d  (%.3f%%)"
                      % (label, 100 * pad, out, len(cloud),
                         100.0 * out / len(cloud)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
