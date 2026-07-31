# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Do per-splat features predict the operator's IN-BOX removals, across datasets?

Usage:
    python scripts/inbox_rule.py NAME RAW.ply CLEAN.ply points3D.txt \
                                [NAME2 RAW2.ply CLEAN2.ply points3D2.txt ...]

One dataset gives the within-dataset table; two or more add the transfer
tests, which are the point.

What this measured on C + D (n=2 datasets, preliminary)
----------------------------------------------------------------
Out-of-box removals are solved by cropping the input cloud. The remaining
19.7-25.2% of the manual work is in-box, at base rates of 0.39% and 1.82%.

  * The feature ORDERING is shared. Of eight features, the informative two
    are the same on both datasets: distance to the nearest face of the
    delivered box (removals hug the boundary) and local density of the
    surveyed cloud (removals stand where the survey is sparse).
  * Absolute thresholds do NOT transfer. A threshold picked at recall 30%
    on one dataset collapses to 1.8% recall or 5.5% precision on the other:
    the scales are dataset-specific even when the ranking is not.
  * Quantile ranks DO transfer, and need no labels: quantiles come from the
    unlabeled model itself. Flagging the worst 5% by mean quantile of
    (face, dens8) catches 51.1% / 53.2% of the in-box removals on the two
    datasets. Precision follows the base rate (4.0% / 19.4%), so this is a
    REVIEW shortlist -- roughly half the in-box handwork concentrated into
    5% of the model -- not an auto-delete.

Features are computed from the raw PLY and the surveyed cloud only, so the
rule is available the moment training ends, with no labels and no tuning.

The removed set is recovered by exact float32 XYZ match between the raw and
cleaned PLYs (the cleaned model is a row-subset of the raw one; verified
exact on both datasets, 5,000,000 rows, no duplicate positions).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleanup_cost import load_cloud  # noqa: E402

MIN_VISIBLE = 0.01              # the same gate the anchor pull uses
DT = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])

FEATURES = ["opacity", "max_axis", "anisotropy", "volume", "dist_x",
            "dens8_x", "lum", "face"]
# Direction: True when the LOW side of the feature is the suspect side.
# (lum was tried in both directions and is uninformative in either; the
# recorded tables used high-lum-is-suspect, kept here for reproducibility.)
LOW_IS_SUSPECT = {"opacity": True, "face": True}
FLAG_FRACTIONS = (0.005, 0.01, 0.02, 0.05)


def read_ply_full(path):
    """memmap the float32 vertex table and pull the named columns."""
    with open(path, "rb") as fh:
        blob = b""
        while b"end_header" not in blob:
            chunk = fh.read(4096)
            if not chunk:
                raise ValueError("no end_header in %s" % path)
            blob += chunk
    head = blob[:blob.index(b"end_header")]
    off = blob.index(b"end_header") + len(b"end_header")
    # Exactly one newline (optional CR before it); a while-loop would eat the
    # first payload byte when it is 0x0A/0x0D. See mpc_tests/test_ply_header.py.
    if blob[off:off + 1] == b"\r":
        off += 1
    if blob[off:off + 1] == b"\n":
        off += 1
    lines = head.split(b"\n")
    n = int([l.split()[-1] for l in lines if l.startswith(b"element vertex")][0])
    props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    m = np.memmap(path, dtype=np.float32, mode="r", offset=off,
                  shape=(n, len(props)))
    col = {p: i for i, p in enumerate(props)}
    out = {
        "xyz": np.ascontiguousarray(m[:, [col["x"], col["y"], col["z"]]]),
        "opacity_raw": np.ascontiguousarray(m[:, col["opacity"]]),
        "scales_raw": np.ascontiguousarray(
            m[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]),
        "f_dc": np.ascontiguousarray(
            m[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]),
    }
    del m
    return out


def _keys(xyz):
    return np.ascontiguousarray(xyz, dtype=np.float32).view(DT).ravel()


def build_subject(name, raw_ply, clean_ply, points3d):
    """Visible in-box splats of the raw model: removed flag + features.

    Cached beside the raw PLY (the inputs it derives from), keyed by the
    raw file's basename.
    """
    cache = os.path.join(os.path.dirname(os.path.abspath(raw_ply)),
                         "_inbox_features_%s.npz"
                         % os.path.splitext(os.path.basename(raw_ply))[0])
    if os.path.isfile(cache):
        d = dict(np.load(cache))
        print("%s: cached %s" % (name, os.path.basename(cache)))
        return d

    from scipy.spatial import cKDTree

    raw = read_ply_full(raw_ply)
    clean = read_ply_full(clean_ply)
    kept = np.isin(_keys(raw["xyz"]), _keys(clean["xyz"]))

    op = 1.0 / (1.0 + np.exp(-raw["opacity_raw"].astype(np.float64)))
    op_clean = 1.0 / (1.0 + np.exp(-clean["opacity_raw"].astype(np.float64)))
    vis_clean = op_clean >= MIN_VISIBLE
    lo = clean["xyz"][vis_clean].min(axis=0)
    hi = clean["xyz"][vis_clean].max(axis=0)

    sel = (op >= MIN_VISIBLE) & np.all((raw["xyz"] >= lo)
                                       & (raw["xyz"] <= hi), axis=1)
    xyz = raw["xyz"][sel]
    scales = np.exp(raw["scales_raw"][sel].astype(np.float64))

    cloud = load_cloud(points3d)
    tree = cKDTree(cloud)
    sub = cloud[::max(1, len(cloud) // 200000)]
    spacing = float(np.median(cKDTree(sub).query(sub, k=2, workers=-1)[0][:, 1]))
    d8 = tree.query(xyz, k=8, workers=-1)[0]

    rel = (xyz - lo) / (hi - lo)
    d = {
        "removed": (~kept[sel]).astype(np.int8),
        "opacity": op[sel],
        "max_axis": scales.max(axis=1),
        "anisotropy": scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-12),
        "volume": scales.prod(axis=1),
        "dist_x": d8[:, 0] / spacing,
        "dens8_x": d8[:, -1] / spacing,
        "lum": (0.2126 * raw["f_dc"][sel][:, 0]
                + 0.7152 * raw["f_dc"][sel][:, 1]
                + 0.0722 * raw["f_dc"][sel][:, 2]).astype(np.float64),
        "face": np.minimum(rel, 1.0 - rel).min(axis=1),
        "spacing": np.array([spacing]),
    }
    np.savez_compressed(cache, **d)
    return d


def suspicion(d, feat):
    v = d[feat].astype(np.float64)
    return -v if LOW_IS_SUSPECT.get(feat, False) else v


def qrank(v):
    """Quantile rank in [0, 1): fraction of the model below this value."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v)) / len(v)
    return r


def precision_at_recall(score, removed, recall):
    pos = score[removed == 1]
    thr = np.quantile(pos, 1.0 - recall)
    sel = score >= thr
    tp = int((removed[sel] == 1).sum())
    return thr, tp / max(int(sel.sum()), 1)


def main(argv):
    if len(argv) < 4 or len(argv) % 4:
        print(__doc__)
        return 2
    datasets = {argv[i]: build_subject(*argv[i:i + 4])
                for i in range(0, len(argv), 4)}

    for name, d in datasets.items():
        removed = d["removed"].astype(bool)
        base = removed.mean()
        print("\n=== %s: %d in-box visible, removed %.3f%% ==="
              % (name, len(removed), 100 * base))
        print("  within-dataset precision at recall 30%:")
        for feat in FEATURES:
            _, prec = precision_at_recall(suspicion(d, feat),
                                          d["removed"], 0.30)
            print("    %-10s %6.2f%%  (x%.1f)" % (feat, 100 * prec,
                                                  prec / base))
        rule = (1.0 - qrank(d["face"]) + qrank(d["dens8_x"])) / 2
        print("  quantile rule (face+dens8, no labels, no tuning):")
        for q in FLAG_FRACTIONS:
            thr = np.quantile(rule, 1.0 - q)
            s = rule >= thr
            tp = int(removed[s].sum())
            print("    flag top %4.1f%%  precision %5.1f%%  recall %5.1f%%"
                  % (100 * q, 100 * tp / max(int(s.sum()), 1),
                     100 * tp / max(int(removed.sum()), 1)))

    names = list(datasets)
    if len(names) > 1:
        print("\n=== absolute-threshold transfer (recall 30%% on A -> B) ===")
        for a in names:
            for b in names:
                if a == b:
                    continue
                da, db = datasets[a], datasets[b]
                base_b = db["removed"].mean()
                print("  %s -> %s (base %.3f%%)" % (a, b, 100 * base_b))
                for feat in FEATURES:
                    thr, _ = precision_at_recall(suspicion(da, feat),
                                                 da["removed"], 0.30)
                    s = suspicion(db, feat) >= thr
                    tp = int((db["removed"][s] == 1).sum())
                    rec = tp / max(int(db["removed"].sum()), 1)
                    prec = tp / max(int(s.sum()), 1)
                    print("    %-10s precision %6.2f%%  recall %5.1f%%"
                          % (feat, 100 * prec, 100 * rec))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
