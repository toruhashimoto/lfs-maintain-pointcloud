# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the drift-stats snapshot schedule.

The plugin cannot rely on the training_end hook: LichtFeld Studio v0.5.1
registers it (`Python hook registered for hook 4`) but never dispatches it
in headless mode, and the embedded interpreter does not run atexit handlers
either. Final drift statistics therefore have to be written from the
post_step hook, periodically, so that the last successful write is the
final result.

Run:  python -m pytest mpc_tests/  (or plain: python mpc_tests/test_stats_policy.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stats_policy import should_snapshot  # noqa: E402


def test_disabled_interval_never_snapshots():
    assert should_snapshot(it=100, last_snapshot_it=-1, interval=0) is False
    assert should_snapshot(it=100, last_snapshot_it=-1, interval=-5) is False


def test_first_snapshot_happens_immediately():
    # An early write is what lets an operator confirm within seconds that the
    # run is wired up, instead of discovering after hours that nothing applied.
    assert should_snapshot(it=1, last_snapshot_it=-1, interval=1000) is True


def test_no_repeat_for_same_iteration():
    assert should_snapshot(it=500, last_snapshot_it=500, interval=1000) is False


def test_waits_for_full_interval():
    assert should_snapshot(it=999, last_snapshot_it=1, interval=1000) is False


def test_fires_exactly_at_interval():
    assert should_snapshot(it=1001, last_snapshot_it=1, interval=1000) is True


def test_fires_when_iterations_were_skipped():
    # The hook can miss iterations (exception path, strategy internals);
    # the schedule must not stall waiting for an exact multiple.
    assert should_snapshot(it=5000, last_snapshot_it=1, interval=1000) is True


def test_ignores_backwards_iteration():
    # Checkpoint resume restarts the counter; never write an older state
    # over a newer one.
    assert should_snapshot(it=10, last_snapshot_it=900, interval=1000) is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
