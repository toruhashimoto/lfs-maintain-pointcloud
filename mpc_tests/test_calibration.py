# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the self-calibrating dead zone.

The dead zone decides whether the leash acts at all. Measured on two
datasets: with the zone above the free-drift p95 the plugin had zero
effect on any percentile; with it at p75 the same dataset moved p90
-25.6% and p95 -38.3%. These tests fix the selection and clipping rules
that put it there.

Run:  python -m pytest mpc_tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import select_control_rows  # noqa: E402


def test_selects_requested_fraction():
    mask = select_control_rows(10000, 0.02, seed=0)
    assert mask.dtype == np.bool_
    assert mask.shape == (10000,)
    assert mask.sum() == 200


def test_deterministic_for_same_seed():
    a = select_control_rows(10000, 0.02, seed=7)
    b = select_control_rows(10000, 0.02, seed=7)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = select_control_rows(10000, 0.02, seed=1)
    b = select_control_rows(10000, 0.02, seed=2)
    assert not np.array_equal(a, b)


def test_zero_fraction_selects_nothing():
    assert select_control_rows(1000, 0.0, seed=0).sum() == 0


def test_empty_cloud_is_safe():
    # capture() can run before any splat exists; must not raise.
    assert select_control_rows(0, 0.02, seed=0).shape == (0,)


def test_tiny_cloud_rounds_to_zero_without_raising():
    # 10 rows * 2% = 0.2 -> 0 controls. A zero-size control set must be
    # representable, not an error: calibration simply never fires.
    assert select_control_rows(10, 0.02, seed=0).sum() == 0


from calibration import (  # noqa: E402
    CONTROL_QUANTILES, calibrate_free_radius, control_quantiles)


def _drift(n=10000, scale=1.0, seed=3):
    """A positive, right-skewed drift sample like the real distribution."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.lognormal(mean=0.0, sigma=0.6, size=n)) * scale


def test_targets_the_requested_quantile():
    d = _drift()
    r = calibrate_free_radius(d, quantile=70.0, r_static=1e9, r_min=0.0,
                              current=0.0)
    assert abs(r - np.percentile(d, 70.0)) < 1e-9


def test_static_rule_is_an_upper_bound():
    # The static rule is what today's default produces. Calibration may only
    # ever tighten relative to it, so an existing validated run cannot get
    # looser as a side effect of turning calibration on.
    d = _drift(scale=100.0)
    r = calibrate_free_radius(d, quantile=70.0, r_static=0.5, r_min=0.0,
                              current=0.0)
    assert r == 0.5


def test_floor_prevents_collapse_to_zero():
    # A dead zone of 0 with a fixed strength is the documented hard-freeze
    # failure mode, so a degenerate control sample must not produce it.
    d = np.zeros(1000)
    r = calibrate_free_radius(d, quantile=70.0, r_static=1.0, r_min=0.01,
                              current=0.0)
    assert r == 0.01


def test_static_bound_wins_over_floor():
    # If the floor were configured above the static bound, the bound still
    # governs: calibration must never widen past today's default.
    d = np.zeros(1000)
    r = calibrate_free_radius(d, quantile=70.0, r_static=0.005, r_min=0.01,
                              current=0.0)
    assert r == 0.005


def test_small_change_is_suppressed():
    # Re-targeting the leash every interval for a sub-percent move would
    # churn the equilibrium for no benefit.
    d = _drift()
    p70 = float(np.percentile(d, 70.0))
    assert calibrate_free_radius(d, 70.0, 1e9, 0.0, current=p70 * 1.02) is None


def test_large_change_is_applied():
    d = _drift()
    p70 = float(np.percentile(d, 70.0))
    r = calibrate_free_radius(d, 70.0, 1e9, 0.0, current=p70 * 2.0)
    assert r is not None and abs(r - p70) < 1e-9


def test_empty_control_sample_keeps_current():
    assert calibrate_free_radius(np.array([]), 70.0, 1.0, 0.0, 0.5) is None


def test_no_static_bound_means_no_calibration():
    # r_static == 0 means nn_spacing was never computed (numpy/scipy missing).
    # Calibrating against an unknown bound is worse than not calibrating.
    assert calibrate_free_radius(_drift(), 70.0, 0.0, 0.0, 0.0) is None


def test_control_quantiles_reports_every_candidate():
    d = _drift()
    q = control_quantiles(d)
    assert set(q) == {"q%g" % v for v in CONTROL_QUANTILES}
    assert abs(q["q70"] - np.percentile(d, 70.0)) < 1e-9
    # Monotone, so a post-hoc sweep over q reads as a curve.
    vals = [q["q%g" % v] for v in CONTROL_QUANTILES]
    assert vals == sorted(vals)


def test_control_quantiles_on_empty_sample():
    q = control_quantiles(np.array([]))
    assert set(q) == {"q%g" % v for v in CONTROL_QUANTILES}
    assert all(v == 0.0 for v in q.values())


from calibration import escape_counts  # noqa: E402


def test_counts_rows_beyond_multiples_of_the_spacing():
    # A count is the tail statistic the maximum should have been: bounded,
    # additive, and describable by a mean and a stdev. Six measured
    # baselines put max at 10.9-93.9 (9x spread, effect/noise 1.0) while
    # every anchored run sat below every baseline -- a separation only an
    # exact rank test could call. A count does not have that pathology.
    d = np.array([0.5, 1.5, 2.5, 3.5, 10.0])
    c = escape_counts(d, unit=1.0)
    assert c["escaped_1x"] == 4      # >1.0
    assert c["escaped_2x"] == 3      # >2.0
    assert c["escaped_4x"] == 1      # >4.0
    assert c["escaped_frac_2x"] == 3 / 5.0


def test_escape_counts_without_a_spacing():
    # nn_spacing is 0 when numpy/scipy were missing at capture. Reporting
    # zeros beats counting against a meaningless threshold.
    c = escape_counts(np.array([1.0, 2.0]), unit=0.0)
    assert c["escaped_1x"] == 0 and c["escaped_frac_2x"] == 0.0


def test_escape_counts_on_empty_sample():
    c = escape_counts(np.array([]), unit=1.0)
    assert c["escaped_1x"] == 0 and c["escaped_frac_2x"] == 0.0


def test_escape_counts_are_json_safe():
    import json
    json.dumps(escape_counts(np.array([0.5, 5.0]), unit=1.0))
