# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Distribution summary for init-row drift.

Kept free of the ``lichtfeld`` import so it can be unit tested without the
host application, the same way ``stats_policy`` is.

Why more than a median: the drift distribution this plugin produces is
bimodal by construction. The dead zone leaves the bulk of the rows entirely
free, so the median tracks the unanchored baseline almost exactly, while the
leash acts only on the tail that has escaped the dead zone. In the
RAMESSES/Chest study the median moved -5.5% and p90 moved -59% -- reading the
median alone says the plugin does nothing. Emitting p50/p75/p90/p95 together
is what makes that shape visible in a run's own output.
"""

_QUANTILES = (50.0, 75.0, 90.0, 95.0)
_QUANTILE_KEYS = ("p50", "p75", "p90", "p95")


def drift_percentiles(d):
    """Summarise a 1-D array of per-row drift distances.

    Returns plain Python floats (the dict is json.dump'd, and numpy scalars
    are not JSON serialisable). ``median`` is retained as an alias of ``p50``
    because existing result files and the README refer to it by that name.

    An empty input yields zeros rather than raising: every original row can
    legitimately be excluded as relocated, and this runs inside the post_step
    hook where an exception would cost the iteration's teleport snapshots.

    numpy is imported here rather than at module scope so that importing this
    module cannot break ``anchor_core``: every numpy use there is function-local
    behind an ImportError guard precisely so the plugin still loads (degraded)
    when the dependency is missing. A module-scope import here would turn that
    into a hard failure to load at all.
    """
    import numpy as np

    d = np.asarray(d).ravel()
    if d.size == 0:
        return {k: 0.0 for k in
                ("mean", "median", "p50", "p75", "p90", "p95", "max")}

    # One sort pass for all four quantiles -- this runs periodically over
    # millions of rows during training.
    q = np.percentile(d, _QUANTILES)
    out = {
        "mean": float(np.mean(d, dtype=np.float64)),
        "median": float(q[0]),
    }
    out.update({k: float(v) for k, v in zip(_QUANTILE_KEYS, q)})
    out["max"] = float(np.max(d))
    return out
