# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Crop the surveyed cloud to the region of interest before training.

Why this and not a plugin feature
---------------------------------
Measured on the C capture, where the delivered model and the raw one both
survived so the operator's manual cleanup could be used as ground truth.
74.8% of everything they removed lay outside the bounding box of what they
kept. Four ways of attacking that, 32 runs:

    input cloud cropped to the box    out-of-box splats -56.0%  (eff/noise 15.5)
    nn anchoring + anchor_new                            -8.6%  (3.2)
    index anchoring + calibration                        -0.3%  (0.1)
    training-side crop box                              -26.1%  (9.6)

The first costs 0.24 dB inside the delivered box. The last costs **3.71 dB**
inside it -- it is implemented (``LFS_MPC_CROP_BOX``) and must not be used;
see the README. Anchoring cannot win here by construction: of the surveyed
points, 0.68% already lie outside the box, and an anchor's whole job is to
hold a splat on its surveyed point, so it pins rows exactly where they are
least wanted.

So the effective measure is not a training-time regulariser at all. It is
deleting those points from the input, which takes 7.1 s for 4.3M points and
outperforms every in-training lever by roughly 6x.

What this does not do
---------------------
It does not renumber POINT3D_IDs, and ``images.txt`` keeps referring to points
that are no longer in ``points3D.txt``. COLMAP's own readers ignore comments
and tolerate dangling references, and LichtFeld Studio only wants the cloud
for initialisation and the cameras for poses -- the cropped C dataset
trained without complaint. Rewriting a 479 MB ``images.txt`` to prune the
references would cost more than it buys. If a future reader rejects them,
that is the thing to revisit first.
"""

import json
import os
import shutil
import subprocess

try:                                        # inside the plugin package
    from .cropbox import inside_box, pad_box, parse_box
except ImportError:                         # standalone, as tests and the CLI use it
    from cropbox import inside_box, pad_box, parse_box

COUNT_PREFIX = b"# Number of points:"
_BUF = 1 << 20

# Binary mode throughout. COLMAP writes LF even on Windows; text mode would
# translate every line to CRLF, changing the size of a 437 MB file for no
# reason and leaving correctness up to how forgiving the reader is.
_OPEN = dict(buffering=_BUF)


def _point_xyz(line):
    """XYZ of a points3D.txt data line, or None if it is not one."""
    parts = line.split(None, 4)
    if len(parts) < 4:
        return None
    return (float(parts[1]), float(parts[2]), float(parts[3]))


def scan_points3d(path, box, sample_every=1):
    """Count how many points survive ``box``, without writing anything.

    Choosing a box means trying several, and each trial would otherwise cost a
    437 MB write. ``sample_every=N`` looks at every Nth data line only --
    the GUI's live count uses it so a 4M-point cloud answers in about a
    second; the returned counts are of the SAMPLE (the fraction is what the
    caller wants, and the export always does the exact full pass).
    """
    kept = total = seen = 0
    with open(os.fspath(path), "rb", **_OPEN) as fh:
        for line in fh:
            if not line.strip() or line.startswith(b"#"):
                continue
            take = (seen % sample_every) == 0
            seen += 1
            if not take:
                continue
            xyz = _point_xyz(line)
            if xyz is None:
                continue
            total += 1
            if inside_box(xyz, box):
                kept += 1
    return kept, total


def read_xyz_sample(path, sample_every=1):
    """XYZ of every Nth data line, as a float64 array.

    Feeds cropbox.fit_box: sampling every 8th point of a 4M cloud keeps the
    percentile fit within a fraction of a millimetre of the full answer.
    """
    import numpy as np

    xs = []
    seen = 0
    with open(os.fspath(path), "rb", **_OPEN) as fh:
        for line in fh:
            if not line.strip() or line.startswith(b"#"):
                continue
            take = (seen % sample_every) == 0
            seen += 1
            if not take:
                continue
            xyz = _point_xyz(line)
            if xyz is not None:
                xs.append(xyz)
    return np.asarray(xs, dtype=np.float64)


def crop_points3d(src, dst, box):
    """Write ``src`` to ``dst`` keeping only points inside ``box``.

    Returns ``(kept, total)``. ``dst=None`` counts only.

    Two passes, because the ``# Number of points:`` header has to be correct
    and is not known until the cloud has been scanned. A stale count is the
    quiet failure mode: the dataset trains fine and every number derived from
    it afterwards is wrong.
    """
    kept, total = scan_points3d(src, box)
    if dst is None:
        return kept, total
    with open(os.fspath(src), "rb", **_OPEN) as fh, \
            open(os.fspath(dst), "wb", **_OPEN) as out:
        for line in fh:
            if line.startswith(b"#"):
                if line.startswith(COUNT_PREFIX):
                    # Preserve any ", mean track length: ..." tail rather than
                    # dropping fields we did not recompute.
                    tail = line[len(COUNT_PREFIX):].split(b",", 1)
                    suffix = b"," + tail[1] if len(tail) > 1 else b"\n"
                    out.write(COUNT_PREFIX + b" %d" % kept + suffix)
                else:
                    out.write(line)
                continue
            if not line.strip():
                continue
            xyz = _point_xyz(line)
            if xyz is not None and inside_box(xyz, box):
                out.write(line)
    return kept, total


# ------------------------------------------------------------- whole dataset

DEFAULT_PAD = 0.05


def _link_file(src, dst):
    """Hardlink ``src`` to ``dst``, copying only if the filesystem refuses.

    images.txt is 479 MB on this capture and identical in every arm of a
    campaign. Hardlinking it kept the cropped C dataset to 0.91 GB of
    additional disk.
    """
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        # Different volume, or a filesystem without hardlinks.
        shutil.copy2(src, dst)
        return "copy"


def _share_dir(src, dst):
    """Point ``dst`` at the same directory as ``src`` without copying it.

    A junction on Windows rather than a symlink: junctions need no privilege,
    symlinks need Developer Mode or an elevated shell.

    normpath is not cosmetic here: mklink is a cmd builtin and rejects
    '/'-separated paths, which is how they arrive from any POSIX-style shell.
    """
    if os.name == "nt":
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", os.path.normpath(dst),
             os.path.normpath(src)], capture_output=True, text=True)
        if done.returncode:
            raise OSError("mklink /J failed for %s -> %s: %s"
                          % (dst, src, (done.stdout + done.stderr).strip()))
        return "junction"
    os.symlink(src, dst, target_is_directory=True)
    return "symlink"


def crop_dataset(src_root, dst_root, box_text, pad=DEFAULT_PAD, dry_run=False):
    """Build a COLMAP dataset at ``dst_root`` whose cloud is cropped to a box.

    ``src_root`` and ``dst_root`` are the directories holding ``sparse/0`` and
    ``images``. ``box_text`` is ``"x0,y0,z0:x1,y1,z1"``; ``pad`` grows it by a
    fraction of each axis' own extent.

    Returns a report dict, also written to ``crop_input.json`` in the output,
    so a campaign of several boxes can be told apart after the fact.
    """
    box = parse_box(box_text)
    if box is None:
        raise ValueError(
            "no crop box given; expected 'x0,y0,z0:x1,y1,z1'. Refusing to "
            "copy the cloud through unchanged, which would look cropped.")
    if pad < 0.0:
        raise ValueError(
            "pad must be >= 0 (got %g); a negative pad silently shrinks the "
            "box, which inverts the margin's whole purpose" % pad)
    box = pad_box(box, pad)

    src_root, dst_root = os.fspath(src_root), os.fspath(dst_root)
    src_sparse = os.path.join(src_root, "sparse", "0")
    src_pts = os.path.join(src_sparse, "points3D.txt")
    if not os.path.exists(src_pts):
        raise FileNotFoundError(src_pts)

    report = {"src": src_root, "dst": dst_root, "box": box, "pad": float(pad),
              "box_requested": box_text}

    if dry_run:
        kept, total = scan_points3d(src_pts, box)
        report.update(kept=kept, total=total, dropped=total - kept,
                      dry_run=True)
        return report

    dst_sparse = os.path.join(dst_root, "sparse", "0")
    dst_pts = os.path.join(dst_sparse, "points3D.txt")
    if os.path.exists(dst_pts):
        raise FileExistsError(
            "%s already exists; remove it rather than half-overwriting a "
            "dataset that a campaign may already have trained on" % dst_pts)
    os.makedirs(dst_sparse, exist_ok=True)

    kept, total = crop_points3d(src_pts, dst_pts, box)
    report.update(kept=kept, total=total, dropped=total - kept, dry_run=False)

    for name in ("cameras.txt", "images.txt"):
        report[name] = _link_file(os.path.join(src_sparse, name),
                                  os.path.join(dst_sparse, name))

    src_images = os.path.join(src_root, "images")
    dst_images = os.path.join(dst_root, "images")
    if os.path.isdir(src_images):
        if not os.path.exists(dst_images):
            report["images"] = _share_dir(src_images, dst_images)
    else:
        # Not fatal here -- some layouts keep images elsewhere -- but a
        # dataset silently missing images/ fails at training launch, which
        # is a worse place to find out.
        report["images"] = "absent"
        print("crop_input: WARNING: %s has no images/ directory; the "
              "cropped dataset will not have one either" % src_root)

    with open(os.path.join(dst_root, "crop_input.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report


# ----------------------------------------------------------------------- CLI


def _glue_box_value(argv):
    """Rewrite ``--box V`` as ``--box=V``.

    Every real box begins with a negative coordinate -- the measured C box
    is ``-0.8777,-1.2579,-1.0:3.1204,0.3682,1.0``. argparse classifies any
    value starting with ``-`` as another option and fails with "expected one
    argument", and its negative-number escape hatch only recognises bare
    numbers, not a comma-separated box. Gluing the pair keeps the obvious
    invocation working alongside the ``--box=`` form.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--box" and i + 1 < len(argv):
            out.append("--box=" + argv[i + 1])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main(argv=None):
    """Prepare a cropped dataset. Returns a process exit status.

    Nonzero on a bad box, so a campaign script cannot carry on to the training
    run believing the dataset was prepared.
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Crop a COLMAP cloud to the region of interest before "
                    "training. Measured on C: -56.0%% out-of-box splats "
                    "for -0.24 dB inside the delivered box.")
    ap.add_argument("--data", required=True,
                    help="COLMAP root holding sparse/0 and images")
    # --box has no environment fallback on purpose: a stale value in a
    # campaign shell would silently supply a box where the operator expects
    # the "no box given" refusal.
    ap.add_argument("--box", default="",
                    help="x0,y0,z0:x1,y1,z1 (corners in either order)")
    ap.add_argument("--pad", type=float, default=DEFAULT_PAD,
                    help="grow the box by this fraction of each axis' extent "
                         "(default %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output root (default: <data>_cropped)")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what the box would drop, write nothing")
    args = ap.parse_args(_glue_box_value(
        list(sys.argv[1:] if argv is None else argv)))

    out = args.out or (os.fspath(args.data).rstrip("\\/") + "_cropped")
    try:
        report = crop_dataset(args.data, out, args.box, pad=args.pad,
                              dry_run=args.dry_run)
    except (ValueError, FileExistsError, FileNotFoundError, OSError) as exc:
        # OSError covers _share_dir's junction failure; without it a mklink
        # problem prints a traceback instead of this one-liner.
        print("crop_input: %s" % exc)
        return 2

    lo, hi = report["box"]
    print("box      lo %.4f %.4f %.4f  hi %.4f %.4f %.4f  (pad %.3f)"
          % (lo + hi + (report["pad"],)))
    print("points   %d of %d kept, %d dropped (%.3f%%)"
          % (report["kept"], report["total"], report["dropped"],
             100.0 * report["dropped"] / max(report["total"], 1)))
    print("output   %s%s" % (out, "  [dry run, nothing written]"
                             if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
