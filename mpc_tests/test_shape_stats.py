# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for splat shape statistics.

Position anchoring has a measured geometric cost the drift numbers do not
show: on RAMESSES/Chest the median max-axis scale inflated +4.6%
(effect/noise 22) and the occupied-volume proxy +5% at the v0.2.0 defaults.
DESIGN.md traces the mechanism to MRNF's long-axis split -- the anchor pulls
the parent back to the pre-split centre while the halved long axis stays
halved, and the unregularised scaling_lr regrows it along that same axis.

That cost has never been measured on the crash dataset, so the statistic has
to exist before the campaign runs.

Tolerances are relative, not absolute: the implementation exponentiates in
float32 to keep the 8M-row array at ~96 MB, which costs ~1e-7 relative
precision on an exp(log(x)) round trip. That is four orders of magnitude
finer than the ~5% effect being measured.

Run:  python -m pytest mpc_tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shape_stats import scale_stats  # noqa: E402


def test_exponentiates_the_raw_log_scale():
    # scaling_raw is a log scale; world-space axis lengths are exp(raw).
    raw = np.zeros((100, 3))          # exp(0) = 1 on every axis
    s = scale_stats(raw)
    assert s["rows"] == 100
    assert np.isclose(s["max_axis_p50"], 1.0, rtol=1e-5)
    assert np.isclose(s["volume_proxy"], 100.0, rtol=1e-5)


def test_takes_the_largest_axis():
    raw = np.log(np.array([[1.0, 2.0, 3.0], [5.0, 1.0, 1.0]]))
    s = scale_stats(raw)
    assert np.isclose(s["max_axis_mean"], 4.0, rtol=1e-5)   # (3 + 5) / 2


def test_volume_proxy_is_the_summed_axis_product():
    raw = np.log(np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]))
    s = scale_stats(raw)
    assert np.isclose(s["volume_proxy"], 6.0 + 8.0, rtol=1e-5)


def test_percentiles_are_ordered():
    rng = np.random.default_rng(0)
    s = scale_stats(rng.normal(size=(10000, 3)))
    assert (s["max_axis_p50"] <= s["max_axis_p75"]
            <= s["max_axis_p90"] <= s["max_axis_p95"])


def test_empty_input_is_safe():
    # Runs inside the snapshot path, where an exception would cost the
    # iteration's teleport snapshots.
    s = scale_stats(np.zeros((0, 3)))
    assert s["rows"] == 0
    assert s["volume_proxy"] == 0.0
    assert s["max_axis_p50"] == 0.0


def test_accepts_flat_input():
    s = scale_stats(np.zeros(300))    # [N*3] flattened
    assert s["rows"] == 100


def test_returns_plain_floats_for_json():
    # The dict is json.dump'd; numpy scalars are not JSON serialisable.
    import json
    json.dumps(scale_stats(np.zeros((10, 3))))
