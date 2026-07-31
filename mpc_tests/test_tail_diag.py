# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the largest-drifter profile.

Two calibrated runs at --max-cap 16000000, identical in every aggregate
(relocations 12.89M vs 12.90M, exclusion 30.3% vs 30.2%, p95 0.003873 vs
0.003857), ended at max drift 16.657 and 1.759. One row differs, so the row
has to be looked at directly. These tests fix what "looking at it" reports.

Run:  python -m pytest mpc_tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tail_diag import tail_diagnostics  # noqa: E402


def _cols(n):
    """drift, opacity, max_axis, residual for n rows, all benign."""
    return (np.arange(n, dtype=np.float64),
            np.full(n, 0.9),
            np.full(n, 0.01),
            np.arange(n, dtype=np.float64))


def test_selects_the_largest_drifters():
    d, op, ax, rs = _cols(100)
    t = tail_diagnostics(d, op, ax, rs, k=5)
    assert t["k"] == 5
    assert [r["drift"] for r in t["top"]] == [99.0, 98.0, 97.0, 96.0, 95.0]
    assert t["drift_min_of_tail"] == 95.0


def test_flags_rows_that_receive_no_pull():
    # The mechanism under test: min_pull_opacity leaves invisible rows
    # entirely unpulled, so they can drift without bound by design.
    d, op, ax, rs = _cols(100)
    op[95:] = 0.0001                     # the top five are near-dead
    t = tail_diagnostics(d, op, ax, rs, k=5, min_pull_opacity=0.01)
    assert t["below_min_pull_opacity"] == 5
    assert t["opacity_p50"] < 0.01


def test_alive_drifters_are_not_flagged():
    d, op, ax, rs = _cols(100)
    t = tail_diagnostics(d, op, ax, rs, k=5, min_pull_opacity=0.01)
    assert t["below_min_pull_opacity"] == 0


def test_detects_rows_whose_anchor_never_moved():
    # residual == drift means the live anchor is still the original one:
    # nothing ever re-anchored this slot.
    d, op, ax, rs = _cols(100)
    t = tail_diagnostics(d, op, ax, rs, k=5)
    assert t["never_reanchored"] == 5


def test_detects_rows_that_were_reanchored():
    d, op, ax, rs = _cols(100)
    rs[95:] = 0.001                      # pulled tight to a NEW anchor
    t = tail_diagnostics(d, op, ax, rs, k=5)
    assert t["never_reanchored"] == 0
    assert t["top"][0]["residual"] == 0.001


def test_k_larger_than_the_population():
    d, op, ax, rs = _cols(3)
    t = tail_diagnostics(d, op, ax, rs, k=32)
    assert t["k"] == 3 and len(t["top"]) == 3


def test_empty_input_is_safe():
    # Runs inside the snapshot path; an exception there costs the
    # iteration's teleport snapshots.
    t = tail_diagnostics(np.array([]), np.array([]), np.array([]),
                         np.array([]))
    assert t["k"] == 0 and t["top"] == []


def test_output_is_json_safe():
    import json
    d, op, ax, rs = _cols(50)
    json.dumps(tail_diagnostics(d, op, ax, rs, k=8))
