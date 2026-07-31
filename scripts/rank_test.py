# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact permutation test for metrics that mean/stdev cannot handle.

Usage:  python scripts/rank_test.py <runs-B>

Why this exists
---------------
Maximum init-row drift resisted every attempt to call it. The effect was
-90% or better in every single run, yet effect/noise never rose above 1.0
even with six baselines, and HANDOFF's estimate that "four more baselines
will settle it" turned out to be wrong.

The reason is that the statistic is not one a mean and a standard deviation
describe. Measured baselines: 10.879, 11.408, 79.971, 41.405, 11.023,
93.861 -- bimodal, a group near 11 and a group from 41 to 94, spanning 9x.
The standard deviation of that is enormous and says nothing about whether
the two arms differ, because the arms are separated in RANK, not in a way
scaled by the baseline's own spread.

A permutation test on ranks ignores the spread entirely. It asks only: if
the arm labels were meaningless, how often would the anchored runs happen
to occupy the lowest positions? With every anchored value below every
baseline value, that probability is 1/C(n_total, n_anchored), which is an
exact p-value requiring no distributional assumption.

This is reported as a one-sided test: the direction (anchoring reduces
drift) was fixed by the mechanism before the data was collected.
"""

import itertools
import json
import os
import sys

ARMS = {
    "P1_base_1": "base_8M", "P1_base_2": "base_8M",
    "P4_base_3": "base_8M", "P4_base_4": "base_8M",
    "P4_base_5": "base_8M", "P4_base_6": "base_8M",
    "P2_calib_1": "calib_8M", "P2_calib_2": "calib_8M",
    "P3_fixed10_1": "fixed10_8M", "P3_fixed10_2": "fixed10_8M",
    "P5_base_hi_1": "base_hi", "P5_base_hi_2": "base_hi",
    "P5_calib_hi_1": "calib_hi", "P5_calib_hi_2": "calib_hi",
}

TESTS = [
    ("max drift, 8M cap: any anchor vs baseline",
     ["base_8M"], ["calib_8M", "fixed10_8M"]),
    ("max drift, 8M cap: calibrated only vs baseline",
     ["base_8M"], ["calib_8M"]),
    ("max drift, high cap: calibrated vs baseline",
     ["base_hi"], ["calib_hi"]),
    ("max drift, all caps pooled",
     ["base_8M", "base_hi"], ["calib_8M", "fixed10_8M", "calib_hi"]),
]


def one_sided_p(control, treated):
    """P(treated ranks this low or lower | labels exchangeable).

    Enumerates every way to split the pooled values into groups of the
    observed sizes and counts how many give a treated-group rank sum at or
    below the observed one. Exact, no distributional assumption; feasible
    because these arms are small.
    """
    pooled = list(control) + list(treated)
    k = len(treated)
    # Rank sum is the statistic: lower means the treated arm sits lower.
    order = sorted(range(len(pooled)), key=lambda i: pooled[i])
    rank = [0] * len(pooled)
    for r, i in enumerate(order):
        rank[i] = r + 1
    observed = sum(rank[len(control):])
    total = 0
    atleast = 0
    for combo in itertools.combinations(range(len(pooled)), k):
        total += 1
        if sum(rank[i] for i in combo) <= observed:
            atleast += 1
    return observed, atleast / float(total), total


def load(root, metric_path):
    vals = {}
    for name, arm in ARMS.items():
        p = os.path.join(root, name, "stats.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        if not j.get("final"):
            continue
        d = j
        for key in metric_path:
            d = (d or {}).get(key)
        if d is not None:
            vals.setdefault(arm, []).append((name, float(d)))
    return vals


def main(root):
    vals = load(root, ["init_drift", "max"])
    print("=" * 74)
    print("MAX INIT-ROW DRIFT -- exact one-sided permutation test on ranks")
    print("=" * 74)
    for arm in sorted(vals):
        xs = sorted(v for _n, v in vals[arm])
        print("  %-12s n=%d  %s" % (arm, len(xs),
                                    " ".join("%.3f" % v for v in xs)))
    print()
    for title, ctrl_arms, treat_arms in TESTS:
        c = [v for a in ctrl_arms for _n, v in vals.get(a, [])]
        t = [v for a in treat_arms for _n, v in vals.get(a, [])]
        if len(c) < 2 or not t:
            continue
        observed, p, total = one_sided_p(c, t)
        sep = max(t) < min(c)
        print(title)
        print("  control n=%d (min %.3f)   treated n=%d (max %.3f)"
              % (len(c), min(c), len(t), max(t)))
        print("  ranges fully separated: %s" % sep)
        print("  rank sum %d, exact one-sided p = %.4f  (1 of %d splits)"
              % (observed, p, total))
        print("  -> %s at alpha=0.05"
              % ("SIGNIFICANT" if p <= 0.05 else "not significant"))
        print()
    print("Compare: the same data under mean/stdev gives effect/noise 1.0,")
    print("because the baseline's own spread (9x, bimodal) swamps a real")
    print("and perfectly consistent separation.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else r"<runs-B>"))
