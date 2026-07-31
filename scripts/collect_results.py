# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collapse a campaign's stats JSONs into one comparison table.

Usage:  python scripts/collect_results.py <runs-B>

Reports the numbers the protocol actually reads, in the order it reads
them: applied_iters first (0 means nothing happened, which has cost this
project a full day of runs before), then the dead zone that was in force,
then the drift percentiles, the control-row free drift, and the shape cost.
"""

import json
import os
import sys


def _g(d, *path):
    """Nested get that tolerates a null sub-object (calibration off)."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


COLUMNS = [
    ("run", lambda name, _j: name),
    ("enabled", lambda _n, j: j.get("enabled")),
    ("calibrate", lambda _n, j: _g(j, "config", "calibrate")),
    ("applied", lambda _n, j: j.get("applied_iters")),
    ("iter", lambda _n, j: j.get("iter")),
    ("final", lambda _n, j: j.get("final")),
    ("free_radius", lambda _n, j: _g(j, "config", "free_radius_effective")),
    ("static_bound", lambda _n, j: (
        (_g(j, "config", "free_radius_spacing") or 0)
        * (_g(j, "config", "nn_spacing") or 0))),
    ("n_calib", lambda _n, j: len(j.get("free_radius_history") or [])),
    ("drift_p50", lambda _n, j: _g(j, "init_drift", "p50")),
    ("drift_p75", lambda _n, j: _g(j, "init_drift", "p75")),
    ("drift_p90", lambda _n, j: _g(j, "init_drift", "p90")),
    ("drift_p95", lambda _n, j: _g(j, "init_drift", "p95")),
    ("drift_max", lambda _n, j: _g(j, "init_drift", "max")),
    ("rows_meas", lambda _n, j: _g(j, "init_drift", "rows_measured")),
    ("ctl_rows", lambda _n, j: _g(j, "control_drift", "rows")),
    ("ctl_p75", lambda _n, j: _g(j, "control_drift", "p75")),
    ("ctl_p90", lambda _n, j: _g(j, "control_drift", "p90")),
    ("ctl_p95", lambda _n, j: _g(j, "control_drift", "p95")),
    ("scale_p50", lambda _n, j: _g(j, "scale", "max_axis_p50")),
    ("scale_p90", lambda _n, j: _g(j, "scale", "max_axis_p90")),
    ("volume", lambda _n, j: _g(j, "scale", "volume_proxy")),
    ("teleports", lambda _n, j: j.get("teleports")),
    ("errors", lambda _n, j: j.get("errors")),
]


def main(root):
    rows = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "stats.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                j = json.load(fh)
        except (OSError, ValueError) as e:
            print("SKIP %s: %r" % (name, e), file=sys.stderr)
            continue
        rows.append([fn(name, j) for _h, fn in COLUMNS])

    lines = ["\t".join(h for h, _ in COLUMNS)]
    for r in rows:
        lines.append("\t".join("" if v is None else str(v) for v in r))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(root, "summary.tsv"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n%d run(s) collected -> %s"
          % (len(rows), os.path.join(root, "summary.tsv")), file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else r"<runs-B>"))
