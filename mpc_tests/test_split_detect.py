# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for long-axis split detection.

The constant is not a guess: it is read straight off
densification_kernels.cu, where the split applies log(0.5) to the longest
axis and log(0.85) to each of the other two. Their sum is the log-volume
change, and it is rotation independent -- unlike "the longest axis halved",
which is ambiguous because after the split that axis may not be the longest
any more.

Run:  python -m pytest mpc_tests/
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from split_detect import (  # noqa: E402
    SPLIT_LOG_VOLUME_DELTA, detect_long_axis_splits)


def test_constant_matches_the_kernel():
    assert math.isclose(SPLIT_LOG_VOLUME_DELTA,
                        math.log(0.5) + 2.0 * math.log(0.85))
    assert math.isclose(SPLIT_LOG_VOLUME_DELTA, -1.0181, abs_tol=1e-4)


def test_detects_a_split_row():
    prev = np.array([0.0, -1.0, 2.0])
    cur = prev.copy()
    cur[1] += SPLIT_LOG_VOLUME_DELTA
    m = detect_long_axis_splits(cur, prev)
    assert list(m) == [False, True, False]


def test_ignores_an_ordinary_optimiser_step():
    # scaling_lr moves log-scale by a tiny amount per iteration; a full nat
    # of log-volume is out of reach.
    prev = np.zeros(5)
    cur = np.array([0.001, -0.002, 0.0005, -0.01, 0.02])
    assert not detect_long_axis_splits(cur, prev).any()


def test_ignores_growth_in_the_other_direction():
    # A row whose volume GREW by the same magnitude is not a split.
    prev = np.zeros(3)
    cur = prev - SPLIT_LOG_VOLUME_DELTA
    assert not detect_long_axis_splits(cur, prev).any()


def test_tolerance_covers_a_concurrent_optimiser_step():
    prev = np.zeros(3)
    cur = np.array([SPLIT_LOG_VOLUME_DELTA + 0.03,
                    SPLIT_LOG_VOLUME_DELTA - 0.03,
                    SPLIT_LOG_VOLUME_DELTA + 0.20])
    m = detect_long_axis_splits(cur, prev, tol=0.05)
    assert list(m) == [True, True, False]


def test_shape_mismatch_after_growth_is_safe():
    # For one iteration after MRNF appends rows the two arrays disagree;
    # this runs inside the post_step hook, where raising costs the
    # iteration's teleport snapshots.
    m = detect_long_axis_splits(np.zeros(10), np.zeros(7))
    assert m.shape == (10,) and not m.any()


def test_empty_input_is_safe():
    assert detect_long_axis_splits(np.array([]), np.array([])).shape == (0,)
