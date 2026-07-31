# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-row profile of the largest drifters.

Why this exists
---------------
At --max-cap 8000000 the leash reliably clamps the extreme tail: every
anchored run came in below every baseline (exact rank test p = 0.0048). At
16000000, where density control actually has room to run, it does not --
two otherwise identical calibrated runs ended at max drift 16.657 and 1.759.

Aggregates cannot say why. Those two runs had the same relocation count
(12.89M vs 12.90M), the same exclusion rate (30.3% vs 30.2%) and the same
p95 (0.003873 vs 0.003857). The difference is one row, so the row itself
has to be looked at. Two mechanisms predict a single escaped row:

  * an MCMC relocation that the teleport detector missed -- the slot keeps
    its original anchor, so the *relocation distance* gets reported as
    drift (the failure mode `_orig` exists to prevent);
  * a row below `min_pull_opacity`, which by design receives no pull at
    all, because it is invisible and MCMC noise moves it freely.

They are told apart by the row's opacity, and by whether its distance from
the live anchor still equals its distance from the original one (i.e.
nothing ever re-anchored it).

Kept free of the ``lichtfeld`` import so it can be unit tested without the
host application, like ``stats_policy`` and ``drift_stats``.
"""

_TOP_ROWS_REPORTED = 8


def tail_diagnostics(drift, opacity, max_axis, residual, k=32,
                     min_pull_opacity=0.01):
    """Profile the ``k`` largest drifters.

    drift    : [N] distance from the ORIGINAL captured position
    opacity  : [N] sigmoid(opacity_raw), i.e. 0..1 visibility
    max_axis : [N] largest world-space axis length of the splat
    residual : [N] distance from the LIVE anchor. Equal to drift for a row
               that was never re-anchored; smaller for one that was.

    All inputs are aligned to the same rows, already filtered to the ones
    the drift statistic measures.
    """
    import numpy as np

    d = np.asarray(drift).ravel()
    out = {"k": 0, "below_min_pull_opacity": 0, "never_reanchored": 0,
           "opacity_p50": 0.0, "drift_min_of_tail": 0.0, "top": []}
    if d.size == 0:
        return out

    k = int(min(max(1, k), d.size))
    # argpartition, not a full sort: this runs over millions of rows inside
    # the snapshot path.
    idx = np.argpartition(d, d.size - k)[d.size - k:]
    idx = idx[np.argsort(d[idx])[::-1]]

    op = np.asarray(opacity).ravel()[idx]
    ax = np.asarray(max_axis).ravel()[idx]
    rs = np.asarray(residual).ravel()[idx]
    dd = d[idx]

    out["k"] = int(k)
    # The discriminator: a row this transparent gets no pull by design.
    out["below_min_pull_opacity"] = int(np.count_nonzero(
        op < min_pull_opacity))
    # Within 1% means the live anchor is still the original one, so no
    # relocation was ever detected for this slot.
    out["never_reanchored"] = int(np.count_nonzero(
        np.abs(rs - dd) <= 0.01 * np.maximum(dd, 1e-12)))
    out["opacity_p50"] = float(np.median(op))
    out["drift_min_of_tail"] = float(dd[-1])
    out["top"] = [
        {"drift": float(dd[i]), "opacity": float(op[i]),
         "max_axis": float(ax[i]), "residual": float(rs[i])}
        for i in range(min(_TOP_ROWS_REPORTED, k))
    ]
    return out
