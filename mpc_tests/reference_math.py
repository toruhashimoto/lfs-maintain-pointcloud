# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-numpy reference of the anchor pull math in anchor_core.apply().

This mirrors the lf.Tensor implementation 1:1 so its semantics can be
unit-tested without an embedded LichtFeld interpreter. Any change to the
pull math or relocation detection in anchor_core.py must be reflected here.
"""

import numpy as np

_EPS = 1e-12
_BIG = 1e30

RELOC_MIN_OPACITY = 0.02
REVIVE_FROM = 0.02
REVIVE_TO = 0.10


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def anchor_pull(means, anchors, mask, strength, free_radius=0.0,
                huber_delta=0.0, max_distance=0.0, opacity=None,
                opacity_gate=False, min_pull_opacity=0.0, control=None):
    """One proximal correction step. Returns the new means array.

    means, anchors : [N,3] float arrays
    mask           : [N,1] float, 1.0 = anchored row
    opacity        : [N,1] raw logits or None
    opacity_gate   : multiply pull by sigmoid(opacity)
    min_pull_opacity : rows with sigmoid(opacity) below this get no pull
    control        : [N,1] float, 1.0 = held out from the pull entirely.
                     These rows carry the free-drift estimate the dead-zone
                     calibration reads, so nothing may ever move them --
                     a pulled control row would make the calibration
                     measure the leash's own output.

    anchor_core folds the control rows into ``mask`` at capture time rather
    than gating separately, so the two multiplies are equivalent there. The
    argument is kept independent here so the invariant stays pinned even if
    the mask's role changes.
    """
    d = means - anchors                                   # [N,3]
    r = np.linalg.norm(d, axis=1, keepdims=True)          # [N,1]
    excess = np.maximum(r - free_radius, 0.0) if free_radius > 0.0 else r

    hi = huber_delta if huber_delta > 0.0 else _BIG
    pull = np.clip(excess, 0.0, hi) * strength
    pull = pull * mask
    if control is not None:
        pull = pull * (1.0 - control)
    if max_distance > 0.0:
        pull = np.where(r <= max_distance, pull, 0.0)
    if opacity is not None:
        sig = _sigmoid(opacity)
        if min_pull_opacity > 0.0:
            pull = pull * (sig >= min_pull_opacity)
        if opacity_gate:
            pull = pull * sig
    coef = pull / np.clip(r, _EPS, _BIG)
    return means - d * coef


def detect_relocations(means, prev, op_raw, prev_op_raw, threshold):
    """[N,1] bool mask of rows considered relocated by MCMC.

    A relocation is (a) a positional jump above the threshold on a row
    that is currently alive (near-dead rows jump from injected noise and
    must not count), or (b) a dead->alive opacity discontinuity that the
    optimizer cannot produce in one step (relocation copies a sampled
    alive opacity).
    """
    jump = np.linalg.norm(means - prev, axis=1, keepdims=True)
    cur_sig = _sigmoid(op_raw)
    prev_sig = _sigmoid(prev_op_raw)
    jumped = jump > threshold
    alive = cur_sig >= RELOC_MIN_OPACITY
    revived = (prev_sig < REVIVE_FROM) & (cur_sig > REVIVE_TO)
    return (jumped & alive) | revived
