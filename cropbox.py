# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Region of interest for training, expressed as an axis-aligned box.

Why a crop box belongs in this plugin
-------------------------------------
Comparing the delivered model against the raw one on the C capture,
74.8% of everything the operator removed by hand lay outside the bounding
box of what they kept -- and not one kept splat lay outside it. That
boundary is exact, unlike the in-box removals, which the operator reports
vary per vehicle and which no per-splat feature predicts (max_axis reaches
4% precision, distance to the surveyed cloud 16%).

Position anchoring barely touches this. Of the 36,999 surveyed rows removed
as out-of-box, only 21,096 drifted out from inside; the other 15,903 were
anchored to surveyed points that already lay outside, where the anchor
holds them. Measured: index-mode anchoring changed the out-of-box count by
-0.3% (effect/noise 0.1). Retargeting to the nearest surveyed point and
pulling the grown rows too (nn + anchor_new) managed -8.6%.

The engine already has the right lever and it is not on the CLI. A crop box
is a scene node; `cropbox_lr_scale` (default 0.1) scales the Adam step of
splats it rejects and `cropbox_loss_weight` (0.1) scales the pixel loss for
rays that miss it, but only those two scales are exposed as parameters. The
box itself is reachable from Python:

    scene.get_or_create_cropbox_for_splat(splat_node_id) -> cropbox_id
    scene.get_cropbox_data(cropbox_id).set("min"/"max"/"enabled", ...)

The vec3 properties take a TUPLE. A list raises RuntimeError('bad cast'),
which is the kind of thing worth writing down.

This module is the pure part: parsing and validating the box. Applying it
needs the live scene and lives in anchor_core.
"""


def parse_box(text):
    """Parse "x0,y0,z0:x1,y1,z1" into ((x0,y0,z0), (x1,y1,z1)).

    Returns None for empty input, so "not configured" and "configured
    wrongly" stay distinguishable -- the second raises.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    halves = text.split(":")
    if len(halves) != 2:
        raise ValueError(
            "crop box must be 'x0,y0,z0:x1,y1,z1', got %r" % text)
    out = []
    for half in halves:
        parts = [p.strip() for p in half.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "each corner needs three components, got %r" % half)
        out.append(tuple(float(p) for p in parts))
    lo, hi = out
    # Tolerate the corners being given in either order rather than failing:
    # an operator typing a box by hand has no reason to know which is which.
    lo, hi = (tuple(min(a, b) for a, b in zip(lo, hi)),
              tuple(max(a, b) for a, b in zip(lo, hi)))
    if any(h - l <= 0.0 for l, h in zip(lo, hi)):
        raise ValueError("crop box has zero extent on some axis: %r" % text)
    return lo, hi


def inside_box(xyz, box):
    """True when ``xyz`` lies within ``box``, boundary included.

    Inclusive on purpose: a box derived from measured extremes has its
    extreme points sitting exactly on the face, and half-open intervals
    would drop them.
    """
    lo, hi = box
    return all(l <= v <= h for v, l, h in zip(xyz, lo, hi))


def box_to_text(box):
    """Inverse of :func:`parse_box`, for display and copy-paste.

    The GUI shows the gizmo box in exactly the form ``crop_input --box``
    accepts, so what the operator sees is what the CLI would do.
    """
    lo, hi = box
    return "%s:%s" % (",".join("%g" % v for v in lo),
                      ",".join("%g" % v for v in hi))


def fit_box(xyz, lo_quantile, hi_quantile):
    """Per-axis percentile box of a point sample.

    The RealityScan-style starting point: fit the region to where the mass
    of the cloud actually is, then let the operator adjust. Percentiles
    rather than min/max because SfM leaves points at near-infinity (the
    D cloud spans 710 km on z; its p0.5-p99.5 spans 2.5 m).
    """
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        raise ValueError("fit_box needs at least one point")
    lo = np.percentile(xyz, lo_quantile, axis=0)
    hi = np.percentile(xyz, hi_quantile, axis=0)
    # A degenerate axis (planar scene, single point) would produce a box
    # parse_box refuses; give it a hair of extent instead.
    eps = 1e-6
    hi = np.where(hi - lo <= eps, lo + eps, hi)
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def pad_box(box, fraction):
    """Grow a box by ``fraction`` of each axis' own extent.

    The delivered box is only known after delivery, so a box configured in
    advance is nominal and wants margin. Measured on C: cropping the
    input cloud to the delivered box exactly would drop 29,175 surveyed
    points, with 5% padding 17,857 -- the padding keeps 11,318 points that
    a tight crop would have thrown away.
    """
    if box is None:
        return None
    lo, hi = box
    pad = tuple((h - l) * float(fraction) for l, h in zip(lo, hi))
    return (tuple(l - p for l, p in zip(lo, pad)),
            tuple(h + p for h, p in zip(hi, pad)))
