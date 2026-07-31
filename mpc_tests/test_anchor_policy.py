# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for when a newly created row starts out anchored.

The combination that motivates this: `nn` mode plus `anchor_new_splats`
is the only way to hold the rows density control creates against a
geometrically meaningful target (the nearest surveyed point) rather than
against their birth position. Measured on the C capture, rows MRNF
appended were 4.3x more likely to be deleted by hand than surveyed rows,
and the ones deleted sat a median 5.16 point spacings off the surface --
so those rows are exactly the ones worth holding.

But `_nn_retarget` runs only every `nn_refresh` iterations. Anchoring a new
row on arrival would pull it toward its birth position until then, which is
the v0.1.0 failure (relocations +48%, splat scale +119%) reintroduced in
small continuous doses.

Run:  python -m pytest mpc_tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor_policy import (  # noqa: E402
    new_row_mask_fill, nn_retarget_enables_rows)


def test_default_leaves_new_rows_free():
    # v0.2.0's default: the plugin holds the PRE-PLACED cloud, and an MCMC
    # birth position carries no geometric meaning.
    assert new_row_mask_fill(False, "index") == 0.0
    assert new_row_mask_fill(False, "nn") == 0.0


def test_index_mode_anchors_new_rows_at_birth():
    # The v0.1.0 behaviour, kept reachable because "freeze the model wherever
    # each splat was born" is a coherent if different goal.
    assert new_row_mask_fill(True, "index") == 1.0


def test_nn_mode_defers_new_rows_until_a_retarget():
    # The point of the module: in nn mode the birth position is NOT the
    # anchor that was asked for, so the row waits for a real one.
    assert new_row_mask_fill(True, "nn") == 0.0


def test_retarget_switches_rows_on_only_when_asked():
    assert nn_retarget_enables_rows(True) is True
    assert nn_retarget_enables_rows(False) is False
