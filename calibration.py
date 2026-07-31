# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-calibrating dead zone for the position anchor.

Why this exists
---------------
The dead zone was derived from the point cloud's own geometry (median
nearest-neighbour spacing x free_radius_spacing). That is a static property,
but whether the leash does anything at all depends on where the *drift*
distribution lands relative to the zone -- a dynamic property that differs
per dataset and grows during training.

Measured, in drift-percentile coordinates rather than spacing multiples:

    dataset B, zone at free-drift p75    -> p90 -25.6%, p95 -38.3%
    RAMESSES,  zone at free-drift ~p57   -> p90 -55%
    dataset B, zone above free-drift p95 -> no effect on any percentile

Both working conditions put the zone in the p55-p75 band; the inert one put
it outside p95. Expressed as a spacing multiple the two datasets disagreed
(2.0 vs 1.0); expressed as a drift percentile they agree. So the zone should
be set from the drift distribution.

That distribution does not exist at capture time and grows ~5x between 1,500
and 30,000 iterations, so it has to be measured *during* the run. We hold out
a small random subset of the initial rows from the pull entirely -- control
rows -- and read the free-drift distribution off them live.

Deliberately free of any ``lichtfeld`` import so it can be unit-tested
without the embedded interpreter, like ``stats_policy`` and ``drift_stats``.
numpy is imported inside each function for the same reason it is in
``drift_stats``: a module-scope import would turn a missing dependency into a
hard failure to load the plugin at all, instead of the degraded-but-loaded
behaviour the rest of the code is careful to preserve.
"""

# Every quantile is recorded on every snapshot while only one is applied, so
# the choice of q becomes post-hoc analysis of runs already on disk instead
# of a sweep that costs one training run per candidate.
CONTROL_QUANTILES = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0, 95.0)


def select_control_rows(n, fraction, seed=0):
    """Boolean mask of rows held out from the pull, shape ``(n,)``.

    Deterministic for a given ``seed`` so that separate runs -- and separate
    arms of an A/B -- hold out the same rows and stay comparable.

    Returns an all-False mask rather than raising when ``n`` is 0 or the
    fraction rounds to zero rows: capture can legitimately run before any
    splat exists, and a cloud too small to spare a control row should simply
    never calibrate.
    """
    import numpy as np

    n = max(0, int(n))
    mask = np.zeros(n, dtype=bool)
    if n == 0 or fraction <= 0.0:
        return mask
    k = int(round(n * min(1.0, float(fraction))))
    if k <= 0:
        return mask
    rng = np.random.default_rng(seed)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def calibrate_free_radius(control_drift, quantile, r_static, r_min,
                          current, rel_tolerance=0.05):
    """New dead-zone radius, or None to keep the current one.

    ``r_static`` is the existing spacing-derived rule and acts as a hard
    upper bound: calibration may only tighten relative to today's default,
    so turning it on cannot loosen a previously validated configuration and
    the absolute fidelity floor is preserved.

    ``r_min`` is a lower bound. A zero dead zone combined with a fixed
    strength is exactly the hard-freeze failure mode the dead zone exists to
    prevent -- the leash tightens monotonically as the effective means
    learning rate decays -- so a degenerate control sample must not produce
    one.

    Returns None (meaning: keep the current radius) when the control sample
    is empty, when ``r_static`` is unavailable, or when the move is smaller
    than ``rel_tolerance`` of the current value.
    """
    import numpy as np

    d = np.asarray(control_drift).ravel()
    if d.size == 0 or r_static <= 0.0:
        return None
    r_target = float(np.percentile(d, quantile))
    # max() first, then min(): the static bound wins over the floor, so a
    # misconfigured floor can never widen the zone past today's default.
    r_new = min(max(r_target, float(r_min)), float(r_static))
    if r_new <= 0.0:
        return None
    if current > 0.0 and abs(r_new - current) / current < rel_tolerance:
        return None
    return r_new


_ESCAPE_MULTIPLES = (1.0, 2.0, 4.0)


def escape_counts(drift, unit):
    """How many rows drifted beyond 1x / 2x / 4x ``unit``.

    The tail statistic the maximum should have been. `max` is one row's
    value: six measured baselines gave 10.879 / 11.023 / 11.408 / 41.405 /
    79.971 / 93.861 -- bimodal, 9x spread -- so effect/noise stayed at 1.0
    even though every anchored run came in below every baseline. Only an
    exact rank test could call that separation (p = 0.0048).

    A count has none of that pathology: it is bounded by the row count and
    aggregates the whole tail instead of its single most extreme member.

    ``unit`` must be the cloud's median nearest-neighbour spacing, NOT the
    dead zone. The dead zone differs per arm by construction -- that is the
    whole point of calibrating it -- so counting against it would compare
    each arm to a different threshold and mean nothing across arms. The
    spacing is a property of the input point cloud, identical in every run
    on a given dataset, and makes the count answer a question worth asking:
    how many splats ended up more than one, two, or four point spacings
    from where they were placed.

    A unit of 0 (numpy/scipy missing at capture, so no spacing was derived)
    yields zeros rather than counting against a meaningless threshold.
    """
    import numpy as np

    d = np.asarray(drift).ravel()
    out = {}
    for m in _ESCAPE_MULTIPLES:
        out["escaped_%gx" % m] = 0
    out["escaped_frac_2x"] = 0.0
    if d.size == 0 or unit <= 0.0:
        return out
    for m in _ESCAPE_MULTIPLES:
        out["escaped_%gx" % m] = int(np.count_nonzero(d > m * unit))
    out["escaped_frac_2x"] = float(out["escaped_2x"]) / float(d.size)
    return out


def control_quantiles(control_drift):
    """Every candidate quantile of the free-drift sample.

    Recorded on every snapshot even though only one is applied, so the
    choice of q can be made afterwards from runs already on disk instead of
    costing one training run per candidate.
    """
    import numpy as np

    d = np.asarray(control_drift).ravel()
    if d.size == 0:
        return {"q%g" % q: 0.0 for q in CONTROL_QUANTILES}
    vals = np.percentile(d, list(CONTROL_QUANTILES))
    return {"q%g" % q: float(v) for q, v in zip(CONTROL_QUANTILES, vals)}
