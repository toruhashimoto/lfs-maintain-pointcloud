# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splat shape statistics: max-axis scale distribution and volume proxy.

Position anchoring costs shape fidelity even where it improves position
fidelity. At the v0.2.0 defaults on RAMESSES/Chest the median max-axis scale
inflated +4.6% (effect/noise 22) and the occupied-volume proxy +5%, while
the drift percentiles improved -- so the drift numbers alone overstate the
result.

DESIGN.md attributes the mechanism to MRNF's long-axis split: the split
displaces the parent by exp(scale_long) * 0.5, whose magnitude (median
0.069, p99 2.54) sits below the teleport threshold (3.53) and so is never
classified as a relocation. The anchor reels the parent back to the
pre-split centre within ~22 iterations while the halved long axis stays
halved, leaving a coverage error that the unregularised scaling_lr regrows
along exactly that axis -- which matches the measured anisotropic inflation.

Kept free of the ``lichtfeld`` import so it can be unit tested without the
host application, like ``stats_policy`` and ``drift_stats``. numpy is
imported inside the function for the same reason it is there: a
module-scope import would turn a missing dependency into a hard failure to
load the plugin rather than the degraded-but-loaded behaviour the rest of
the code preserves.
"""

_QUANTILES = (50.0, 75.0, 90.0, 95.0)
_KEYS = ("max_axis_p50", "max_axis_p75", "max_axis_p90", "max_axis_p95")


def scale_stats(scaling_raw):
    """Summarise ``scaling_raw`` ([N,3] log scale) as world-space extents.

    Returns plain Python floats (the dict is json.dump'd, and numpy scalars
    are not JSON serialisable).

    Zero rows yields zeros rather than raising: this runs inside the stats
    snapshot path, where an exception would cost the iteration's teleport
    snapshots.
    """
    import numpy as np

    s = np.asarray(scaling_raw).reshape(-1, 3)
    if s.shape[0] == 0:
        out = {"rows": 0, "max_axis_mean": 0.0, "volume_proxy": 0.0}
        out.update({k: 0.0 for k in _KEYS})
        return out

    # float32 keeps the 8M-row exponentiation at ~96 MB rather than ~190 MB;
    # the resulting ~1e-7 relative error is four orders of magnitude finer
    # than the ~5% effect being measured. The volume sum is accumulated in
    # float64 because it runs over millions of terms.
    axes = np.exp(s.astype(np.float32))
    max_axis = axes.max(axis=1)
    q = np.percentile(max_axis, _QUANTILES)
    out = {
        "rows": int(s.shape[0]),
        "max_axis_mean": float(np.mean(max_axis, dtype=np.float64)),
        "volume_proxy": float(np.sum(axes.prod(axis=1), dtype=np.float64)),
    }
    out.update({k: float(v) for k, v in zip(_KEYS, q)})
    return out
