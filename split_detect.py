# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detect MRNF's long-axis split, which the anchor otherwise fights.

The mechanism (DESIGN.md, confirmed against
densification_kernels.cu:long_axis_split_gaussians_inplace_kernel):

    new_scale[longest] = scale[longest] + log(0.5)
    new_scale[other]   = scale[other]   + log(0.85)   (both other axes)
    positions[src]    += R[longest] * exp(scale[longest]) * 0.5

The parent is displaced along its long axis and its long axis is halved; a
child is appended at the mirrored offset. The displacement is far below the
teleport threshold (measured here: splat scales put it near 0.0007-0.003
against a threshold of 0.057), so the relocation detector never sees it.
The anchor then reels the parent back to the pre-split centre while the
halved axis stays halved -- the pair no longer tiles the extent it was
split to cover, and the unregularised scaling_lr regrows that same axis.
That is the documented +4.6% median scale inflation on RAMESSES/Chest, and
the monotone `scale_p90` growth measured here as the leash is tightened
(+2.1% at q70, +2.9% at q60, +3.4% at q50).

Detection uses the SUM of the three log scales rather than the longest
axis. After the split the previously-longest axis may no longer be the
longest, so tracking "the max axis halved" is ambiguous; the sum is not,
and it is rotation independent:

    delta = log(0.5) + 2*log(0.85) = -1.01815

Adam cannot move log-volume by a full nat in one step at any plausible
scaling learning rate, so this is as unambiguous a signature as the
dead->alive opacity discontinuity the relocation detector already relies on.

Kept free of the ``lichtfeld`` import so it can be unit tested without the
host application. ``anchor_core`` mirrors ``detect_long_axis_splits`` with
lf.Tensor operations rather than calling it: the test runs every iteration
over every row, and pulling [N,1] to the host each step would cost ~32 MB
of transfer per iteration at 8M rows. This module is the reference the
tensor version must agree with -- the same relationship
``mpc_tests/reference_math.py`` has with the pull math -- and it is the
single source of the constant, which anchor_core imports.
"""

import math

# log(0.5) + 2*log(0.85): the exact log-volume change one split applies.
SPLIT_LOG_VOLUME_DELTA = math.log(0.5) + 2.0 * math.log(0.85)

# Opacity is multiplied by 0.6 in the same kernel. Not used for detection --
# the volume signature alone is unambiguous -- but recorded here because it
# is the other half of the split's fingerprint.
SPLIT_OPACITY_FACTOR = 0.6


def detect_long_axis_splits(scale_sum, prev_scale_sum, tol=0.05):
    """Boolean mask of rows split along their long axis this iteration.

    ``scale_sum`` is the per-row sum of the three raw (log) scales. A split
    drops it by exactly ``SPLIT_LOG_VOLUME_DELTA``; ``tol`` is the absolute
    slack allowed, which also has to cover the optimiser step applied in the
    same iteration.

    Returns an all-False mask when the two inputs disagree in shape, which
    happens for one iteration after growth appends rows.
    """
    import numpy as np

    cur = np.asarray(scale_sum).ravel()
    prev = np.asarray(prev_scale_sum).ravel()
    if cur.size == 0 or cur.shape != prev.shape:
        return np.zeros(cur.shape, dtype=bool)
    return np.abs((cur - prev) - SPLIT_LOG_VOLUME_DELTA) <= tol
