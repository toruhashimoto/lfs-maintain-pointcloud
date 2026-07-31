# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the training region of interest.

The box is where most of the manual cleanup lives: 74.8% of what the
operator removed on the C capture lay outside the bounding box of what
they kept, and the boundary was exact. Getting the parse wrong would either
crop away real geometry or silently do nothing, so it is pinned here.

Run:  python -m pytest mpc_tests/
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cropbox import pad_box, parse_box  # noqa: E402


def test_parses_two_corners():
    lo, hi = parse_box("-1,-2,-3:4,5,6")
    assert lo == (-1.0, -2.0, -3.0)
    assert hi == (4.0, 5.0, 6.0)


def test_tolerates_whitespace():
    assert parse_box(" -1 , -2 , -3 : 4 , 5 , 6 ") == ((-1.0, -2.0, -3.0),
                                                       (4.0, 5.0, 6.0))


def test_corners_may_be_given_in_either_order():
    # Someone typing a box by hand has no reason to know which corner the
    # code calls "min"; swapping them should not silently produce an
    # inverted box that rejects everything.
    a = parse_box("4,5,6:-1,-2,-3")
    b = parse_box("-1,-2,-3:4,5,6")
    assert a == b


def test_empty_means_not_configured():
    assert parse_box("") is None
    assert parse_box("   ") is None
    assert parse_box(None) is None


def test_malformed_input_raises_rather_than_defaulting():
    # "configured wrongly" must not look like "not configured": a silently
    # ignored box would leave the operator believing the crop was applied.
    for bad in ("1,2,3", "1,2:3,4", "a,b,c:1,2,3", "1,2,3:4,5", "1,2,3:4,5,6:7"):
        with pytest.raises(ValueError):
            parse_box(bad)


def test_zero_extent_raises():
    with pytest.raises(ValueError):
        parse_box("1,2,3:1,5,6")


def test_padding_grows_each_axis_by_its_own_extent():
    lo, hi = pad_box(((0.0, 0.0, 0.0), (10.0, 2.0, 4.0)), 0.05)
    assert lo == pytest.approx((-0.5, -0.1, -0.2))
    assert hi == pytest.approx((10.5, 2.1, 4.2))


def test_padding_none_stays_none():
    assert pad_box(None, 0.05) is None


def test_zero_padding_is_identity():
    box = ((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0))
    assert pad_box(box, 0.0) == box


def test_box_to_text_roundtrips_through_parse_box():
    # The panel shows the gizmo box as text the operator can paste into
    # crop_input --box; a formatting/parsing mismatch would make the GUI
    # and CLI silently disagree about the same box.
    from cropbox import box_to_text
    box = ((-1.2345, -1.6, -1.1), (3.2, 0.60001, 1.1))
    assert parse_box(box_to_text(box)) == box


def test_fit_box_takes_percentiles_per_axis():
    import numpy as np
    from cropbox import fit_box
    xyz = np.zeros((101, 3))
    xyz[:, 0] = np.arange(101)            # 0..100
    xyz[:, 1] = np.arange(101) * 2        # 0..200
    xyz[:, 2] = 5.0
    lo, hi = fit_box(xyz, 1.0, 99.0)
    assert lo[0] == pytest.approx(1.0) and hi[0] == pytest.approx(99.0)
    assert lo[1] == pytest.approx(2.0) and hi[1] == pytest.approx(198.0)
    # A constant axis must still yield a usable (nonzero-extent) box rather
    # than one parse_box would reject.
    assert hi[2] > lo[2]


def test_fit_box_rejects_empty_input():
    import numpy as np
    from cropbox import fit_box
    with pytest.raises(ValueError):
        fit_box(np.zeros((0, 3)), 1.0, 99.0)
