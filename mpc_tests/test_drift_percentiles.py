# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the init-drift distribution summary.

The README tells the operator to read p50/p75/p90/p95 together, because the
drift distribution is bimodal and the median hides it: in the RAMESSES/Chest
study the headline result was p90 -59% while the median moved -5.5%. The
stats writer only emitted mean/median/p95/max, so that headline metric could
not be reproduced from a run's output at all.

Run:  python -m pytest mpc_tests/  (or plain: python mpc_tests/test_drift_percentiles.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_stats import drift_percentiles  # noqa: E402


def test_known_uniform_distribution():
    # 0..100 inclusive: the k-th percentile is exactly k.
    d = np.arange(101, dtype=np.float64)
    s = drift_percentiles(d)
    assert s["p50"] == 50.0
    assert s["p75"] == 75.0
    assert s["p90"] == 90.0
    assert s["p95"] == 95.0
    assert s["max"] == 100.0
    assert s["mean"] == 50.0


def test_median_is_an_alias_of_p50():
    # `median` predates p50 and is what existing result files and the README
    # refer to; dropping or diverging it would silently break comparisons.
    d = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    s = drift_percentiles(d)
    assert s["median"] == s["p50"]


def test_percentiles_are_monotonic():
    rng = np.random.default_rng(12345)
    d = rng.lognormal(mean=0.0, sigma=2.0, size=10000)
    s = drift_percentiles(d)
    assert s["p50"] <= s["p75"] <= s["p90"] <= s["p95"] <= s["max"]


def test_bimodal_tail_is_visible_in_p90_but_not_median():
    # The distribution shape this metric exists to expose: a large frozen
    # bulk plus a small runaway tail. The median cannot see the tail.
    d = np.concatenate([np.full(9000, 0.001), np.full(1000, 5.0)])
    s = drift_percentiles(d)
    assert s["median"] == 0.001
    assert s["p95"] == 5.0
    assert s["p90"] > 0.001      # p90 sits inside the tail, median does not


def test_single_row():
    s = drift_percentiles(np.array([2.5]))
    for k in ("mean", "median", "p50", "p75", "p90", "p95", "max"):
        assert s[k] == 2.5


def test_empty_input_returns_zeros_not_an_exception():
    # Every original row can be excluded as relocated; the stats writer must
    # still produce a file rather than raising inside the post_step hook.
    s = drift_percentiles(np.array([], dtype=np.float64))
    for k in ("mean", "median", "p50", "p75", "p90", "p95", "max"):
        assert s[k] == 0.0


def test_all_values_are_plain_floats():
    # The dict is json.dump'd; numpy scalars are not JSON serialisable.
    s = drift_percentiles(np.arange(10, dtype=np.float32))
    for k, v in s.items():
        assert type(v) is float, f"{k} is {type(v)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


def test_reports_bounded_tail_percentiles():
    # The maximum is one row's value and is useless for inference: measured
    # baselines were 10.879 / 11.023 / 11.408 / 41.405 / 79.971 / 93.861 --
    # bimodal, 9x spread, effect/noise 1.0 even with six runs. p99/p99.9 are
    # bounded versions of the same tail that a mean and stdev can describe.
    d = np.arange(1001, dtype=np.float64)   # 0..1000, k-th pct is k*10
    s = drift_percentiles(d)
    # np.percentile interpolates, so p99.9 lands on 999.0000000000001 --
    # compare with a tolerance rather than pinning the float.
    assert np.isclose(s["p99"], 990.0)
    assert np.isclose(s["p999"], 999.0)


def test_tail_percentiles_stay_monotonic():
    rng = np.random.default_rng(7)
    d = rng.lognormal(mean=0.0, sigma=2.0, size=100000)
    s = drift_percentiles(d)
    assert s["p95"] <= s["p99"] <= s["p999"] <= s["max"]


def test_tail_percentiles_present_on_empty_and_single():
    for d in (np.array([], dtype=np.float64), np.array([2.5])):
        s = drift_percentiles(d)
        assert "p99" in s and "p999" in s
