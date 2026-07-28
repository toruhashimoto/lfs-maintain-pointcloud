# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduling policy for periodic drift-stats snapshots.

Deliberately free of any ``lichtfeld`` import so it can be unit-tested
without the embedded interpreter.
"""


def should_snapshot(it, last_snapshot_it, interval):
    """Whether to write a drift-stats snapshot at iteration ``it``.

    ``interval <= 0`` disables snapshots. The first eligible iteration always
    writes, so an operator can confirm the run is wired up without waiting for
    the first interval to elapse. Iterations at or before the last snapshot
    never write, which keeps a checkpoint resume from overwriting a newer
    state with an older one.
    """
    if interval <= 0:
        return False
    if it <= last_snapshot_it:
        return False
    if last_snapshot_it < 0:
        return True
    return it - last_snapshot_it >= interval
