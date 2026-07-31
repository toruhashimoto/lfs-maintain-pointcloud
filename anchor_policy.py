# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whether a freshly appended or relocated row starts out anchored.

The subtlety this exists for
----------------------------
`anchor_new_splats` means "hold the rows MCMC creates as well as the ones
that came from the survey". In `index` mode the only anchor such a row can
be given is its BIRTH position, and that is the setting v0.1.0 shipped and
v0.2.0 reverted: measured on a real 30k run, only 219,198 of 5,000,000 rows
were still anchored to an init point, so 95.6% of the pull was acting on
arbitrary MCMC sample positions -- relocations rose 48% and splat scale
inflated 119%.

In `nn` mode the anchor is instead the nearest point of a fixed reference
cloud, which is a geometrically meaningful target and exactly what is wanted
for grown rows. But `_nn_retarget` only runs every `nn_refresh` iterations,
so a row appended between refreshes would spend up to that many iterations
being pulled toward its birth position -- reproducing the v0.1.0 failure in
miniature, continuously, for every cohort of new rows.

So: in nn mode a new row starts UNANCHORED regardless of
`anchor_new_splats`, and the next retarget switches it on once it has a real
reference target. Keeping this decision in one tested function rather than
inline is what makes the two modes' semantics inspectable.

Deliberately free of any ``lichtfeld`` import so it can be unit-tested
without the embedded interpreter.
"""


def new_row_mask_fill(anchor_new_splats, mode):
    """1.0 if a newly appended/relocated row should be pulled immediately.

    Returns 0.0 in nn mode even when anchor_new_splats is set, because the
    row's anchor is its birth position until the next retarget.
    """
    if not anchor_new_splats:
        return 0.0
    if mode == "nn":
        return 0.0
    return 1.0


def nn_retarget_enables_rows(anchor_new_splats):
    """Whether a retarget should switch every row's pull on.

    In nn mode every row has a valid reference anchor after a retarget, so
    with anchor_new_splats set there is no longer a reason to exempt the
    grown ones -- which is the whole point of pairing the two.
    """
    return bool(anchor_new_splats)
