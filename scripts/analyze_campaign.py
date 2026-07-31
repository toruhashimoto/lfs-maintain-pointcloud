# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""A/B analysis of a campaign, in the form the project's docs already use.

Usage:  python scripts/analyze_campaign.py <runs-B>

Reports, per metric: each arm's mean, the within-arm spread, the effect,
and effect/noise. The discipline this follows is documented in
docs/HANDOFF.md and README.md:

  * n=1 vs n=1 says nothing. Every arm needs at least 2 runs; an arm with
    one run is reported but its effect/noise is withheld rather than
    printed as if it meant something.
  * "noise" is the LARGER of the two arms' within-arm spreads, matching how
    the published tables were computed (baseline max drift 21.47/41.71 ->
    spread 20.24; anchor 1.99/2.85 -> 0.86; effect 29.17 was divided by
    20.24 to give the reported 1.4).
  * Within-arm spread is |a-b| for n=2 and the sample standard deviation
    for n>2.
  * Anything with effect/noise below ~2 is not a result. The project has
    twice been misled by single-run differences that vanished on replication.
"""

import csv
import json
import os
import sys

# Baselines from P1 and P4 are the same condition; they were split across
# phases only so the highest-value runs land first if the night is cut short.
ARMS = {
    "P1_base_1": "base_8M", "P1_base_2": "base_8M",
    "P4_base_3": "base_8M", "P4_base_4": "base_8M",
    "P4_base_5": "base_8M", "P4_base_6": "base_8M",
    # Same condition as the P1/P4 baselines, re-run on the build that emits
    # the bounded tail statistics. Pooled into the same arm: the shared
    # metrics get n=8, the tail metrics n=2, and each is spread-checked on
    # the runs that actually carry it.
    "T_base_1": "base_8M", "T_base_2": "base_8M",
    "P2_calib_1": "calib_8M", "P2_calib_2": "calib_8M",
    "P3_fixed10_1": "fixed10_8M", "P3_fixed10_2": "fixed10_8M",
    "P5_base_hi_1": "base_hi", "P5_base_hi_2": "base_hi",
    "P5_calib_hi_1": "calib_hi", "P5_calib_hi_2": "calib_hi",
    # Follow-up plan. All compared against the same base_8M baselines.
    "F_q60_1": "calib_q60", "F_q60_2": "calib_q60",
    "F_q50_1": "calib_q50", "F_q50_2": "calib_q50",
    "F_start5k_1": "calib_start5k", "F_start5k_2": "calib_start5k",
    "F_frac10_1": "calib_frac10", "F_frac10_2": "calib_frac10",
    # C (dataset C, lfs_runs_datac). M_index_1/2 run calibration at the
    # default q=70, so they ARE the q70 arm of the sweep; the nn/crop/box
    # arms are a different mechanism and stay unmapped on purpose.
    "M_base_1": "datac_base", "M_base_2": "datac_base",
    "M_index_1": "datac_q70", "M_index_2": "datac_q70",
    "M_q60_1": "datac_q60", "M_q60_2": "datac_q60",
    "M_q50_1": "datac_q50", "M_q50_2": "datac_q50",
}

# (label, extractor). Drift and shape come from the plugin's stats JSON;
# psnr/ssim from the trainer's own metrics.csv (last row = final eval).
METRICS = [
    ("drift_p50", lambda s, m: _g(s, "init_drift", "p50")),
    ("drift_p75", lambda s, m: _g(s, "init_drift", "p75")),
    ("drift_p90", lambda s, m: _g(s, "init_drift", "p90")),
    ("drift_p95", lambda s, m: _g(s, "init_drift", "p95")),
    ("drift_p99", lambda s, m: _g(s, "init_drift", "p99")),
    ("drift_p999", lambda s, m: _g(s, "init_drift", "p999")),
    # Bounded tail statistics. drift_max is kept for continuity with the
    # published tables but cannot be read here: it is one row's value, and
    # six baselines spanning 9x keep effect/noise at 1.0 no matter how many
    # runs are added. Use scripts/rank_test.py for it instead.
    ("escaped_2x", lambda s, m: _g(s, "init_drift", "escaped_2x")),
    ("drift_max", lambda s, m: _g(s, "init_drift", "max")),
    ("scale_p50", lambda s, m: _g(s, "scale", "max_axis_p50")),
    ("scale_p90", lambda s, m: _g(s, "scale", "max_axis_p90")),
    ("volume", lambda s, m: _g(s, "scale", "volume_proxy")),
    ("teleports", lambda s, m: s.get("teleports")),
    # LichtFeld's default eval_steps are {7000, 30000}, so a full run scores
    # both for free. Worth reporting separately: whether quality peaks at
    # 7k and decays by 30k is an open question in HANDOFF -- it did on
    # RAMESSES/Chest (-0.86 dB) and did the opposite on this dataset.
    ("psnr_7k", lambda s, m: m.get("psnr_first")),
    ("ssim_7k", lambda s, m: m.get("ssim_first")),
    ("psnr", lambda s, m: m.get("psnr")),
    ("ssim", lambda s, m: m.get("ssim")),
]

# Comparisons to print, as (label, arm_a, arm_b). Effects are b - a.
COMPARISONS = [
    ("calibration vs baseline (8M)", "base_8M", "calib_8M"),
    ("fixed spacing=1.0 vs baseline (8M)", "base_8M", "fixed10_8M"),
    ("calibration vs fixed spacing=1.0", "fixed10_8M", "calib_8M"),
    ("calibration vs baseline (high cap)", "base_hi", "calib_hi"),
    # q sweep: q70 already lands on the value a manual sweep found, and
    # neither showed a photometric cost. These go tighter to find where one
    # appears -- q60 picks 0.86x and q50 0.74x of that value.
    ("q60 vs baseline", "base_8M", "calib_q60"),
    ("q50 vs baseline", "base_8M", "calib_q50"),
    ("q60 vs q70", "calib_8M", "calib_q60"),
    ("q50 vs q70", "calib_8M", "calib_q50"),
    # Does waiting for a more mature drift distribution beat clamping early?
    ("calibrate_start 5000 vs 1000", "calib_8M", "calib_start5k"),
    # Does holding out 5x more rows shrink the control-row bias?
    ("control_fraction 0.10 vs 0.02", "calib_8M", "calib_frac10"),
    # The same sweep on C: does dataset B's "tighter is monotonically
    # better for position, cost lands in shape" transfer to a capture with
    # 4x the points?
    ("C q70 vs baseline", "datac_base", "datac_q70"),
    ("C q60 vs baseline", "datac_base", "datac_q60"),
    ("C q50 vs baseline", "datac_base", "datac_q50"),
    ("C q60 vs q70", "datac_q70", "datac_q60"),
    ("C q50 vs q70", "datac_q70", "datac_q50"),
]


def _g(d, *path):
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _mean(xs):
    return sum(xs) / float(len(xs))


def _spread(xs):
    """Within-arm spread: |a-b| for n=2, sample stdev for n>2."""
    if len(xs) < 2:
        return None
    if len(xs) == 2:
        return abs(xs[0] - xs[1])
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)) ** 0.5


def _read_metrics_csv(path):
    """Eval rows of the trainer's metrics.csv, as floats.

    Bare keys come from the last row (the 30,000-iteration eval); `_first`
    keys come from the first (7,000). A run shorter than 7,000 iterations
    hits neither of LichtFeld's default eval_steps and yields an empty
    file -- header only, no rows.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    if not rows:
        return {}
    out = {}
    for k, v in rows[-1].items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    if len(rows) > 1:
        for k, v in rows[0].items():
            try:
                out[k + "_first"] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def load(root):
    """{arm: {metric: [values]}} plus a per-run note list."""
    arms, notes = {}, []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        stats_path = os.path.join(run_dir, "stats.json")
        if not os.path.isfile(stats_path):
            continue
        arm = ARMS.get(name)
        if arm is None:
            notes.append("%s: not part of any arm, skipped" % name)
            continue
        try:
            with open(stats_path, "r", encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError) as e:
            notes.append("%s: unreadable stats.json (%r)" % (name, e))
            continue
        if not s.get("final"):
            # Almost always a run still in flight: the plugin snapshots every
            # 1000 iterations so a partial file exists long before the end.
            # Including it silently mixes a mid-training distribution into an
            # end-of-training arm, which inflates the within-arm spread and
            # buries every real effect under it.
            notes.append("%s: EXCLUDED -- stats are from iter %s, not the "
                         "training end (run still in flight?)"
                         % (name, s.get("iter")))
            continue
        if s.get("enabled") and not s.get("applied_iters"):
            notes.append("%s: enabled but applied_iters=0 -- the anchor did "
                         "NOTHING; exclude this run" % name)
            continue
        if not _g(s, "init_drift", "baseline_valid"):
            notes.append("%s: baseline_valid=false (anchors re-captured "
                         "mid-run); drift is not measured from training start"
                         % name)
        m = _read_metrics_csv(os.path.join(run_dir, "metrics.csv"))
        bucket = arms.setdefault(arm, {})
        for label, fn in METRICS:
            v = fn(s, m)
            if v is not None:
                bucket.setdefault(label, []).append(float(v))
    return arms, notes


# Which unanchored baseline each calibrated arm must be compared against.
# Getting this wrong makes the high-cap arm look like it has a 32% bias when
# most of that is the regime: un-saturating density control widens the free
# drift by 26% on its own, before any control-row effect.
BIAS_BASELINE = {
    "calib_8M": "base_8M",
    "calib_q60": "base_8M",
    "calib_q50": "base_8M",
    "calib_start5k": "base_8M",
    "calib_frac10": "base_8M",
    "calib_hi": "base_hi",
    "datac_q70": "datac_base",
    "datac_q60": "datac_base",
    "datac_q50": "datac_base",
}


def bias_report(root):
    """How far the control rows over-report free drift.

    Control rows are meant to be a live sample of what the drift would have
    been without the leash. They are not quite that: they sit in a scene
    where 98% of the rows ARE held back, so they absorb photometric residual
    the anchored rows no longer take and move further than a genuinely
    unanchored run does.

    This compares each calibrated arm's control_drift against the real
    unanchored baseline's init_drift -- the only way to see the bias, since
    it is a comparison across two different metrics rather than two arms.
    """
    QS = ("p50", "p75", "p90", "p95")
    base, ctl = {}, {}
    for name, arm in ARMS.items():
        p = os.path.join(root, name, "stats.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        if not s.get("final"):
            continue
        if arm.startswith("base"):
            for q in QS:
                v = _g(s, "init_drift", q)
                if v is not None:
                    base.setdefault(arm, {}).setdefault(q, []).append(float(v))
        cd = s.get("control_drift")
        if cd:
            for q in QS:
                if cd.get(q) is not None:
                    ctl.setdefault(arm, {}).setdefault(q, []).append(
                        float(cd[q]))
    if not base or not ctl:
        return
    print("\n" + "=" * 78)
    print("CONTROL-ROW BIAS   (control rows vs their own regime's baseline)")
    print("=" * 78)
    print("%-16s %-9s %4s %13s %13s %9s %8s"
          % ("arm", "vs", "q", "baseline", "control", "bias", "x spread"))
    for arm in sorted(ctl):
        b_arm = BIAS_BASELINE.get(arm)
        if b_arm is None or b_arm not in base:
            print("%-16s %-9s  (no matching baseline arm; skipped)"
                  % (arm, b_arm or "?"))
            continue
        for q in QS:
            if q not in base[b_arm] or q not in ctl[arm]:
                continue
            mb, mc = _mean(base[b_arm][q]), _mean(ctl[arm][q])
            sb = _spread(base[b_arm][q])
            rel = 100.0 * (mc - mb) / mb if mb else 0.0
            xs = (abs(mc - mb) / sb) if sb else float("inf")
            print("%-16s %-9s %4s %13.7f %13.7f %+8.2f%% %8.0f"
                  % (arm, b_arm, q, mb, mc, rel, xs))
    print("\n'x spread' is the bias in units of the baseline's own within-arm")
    print("spread. Large values mean the bias is real, not sampling noise.")


def main(root):
    arms, notes = load(root)
    if not arms:
        print("no runs found under %s" % root, file=sys.stderr)
        return 1

    print("=" * 78)
    print("ARMS")
    print("=" * 78)
    for arm in sorted(arms):
        n = max(len(v) for v in arms[arm].values())
        print("  %-14s n=%d" % (arm, n))
    if notes:
        print("\nNOTES")
        for n in notes:
            print("  ! " + n)

    for title, a, b in COMPARISONS:
        if a not in arms or b not in arms:
            continue
        print("\n" + "=" * 78)
        print(title + "   (%s -> %s)" % (a, b))
        print("=" * 78)
        print("%-11s %13s %13s %11s %10s %8s"
              % ("metric", a, b, "effect", "noise", "eff/noise"))
        for label, _fn in METRICS:
            xa, xb = arms[a].get(label), arms[b].get(label)
            if not xa or not xb:
                continue
            ma, mb = _mean(xa), _mean(xb)
            eff = mb - ma
            pct = (" (%+.1f%%)" % (100.0 * eff / ma)) if ma else ""
            sa, sb = _spread(xa), _spread(xb)
            if sa is None or sb is None:
                print("%-11s %13.6g %13.6g %11.4g %10s %8s"
                      % (label, ma, mb, eff, "n<2", "--"))
                continue
            noise = max(sa, sb)
            ratio = abs(eff) / noise if noise > 0 else float("inf")
            print("%-11s %13.6g %13.6g %11.4g %10.4g %8.1f%s"
                  % (label, ma, mb, eff, noise, ratio, pct))
    bias_report(root)
    print("\nEffect/noise below ~2 is not a result.")
    print("drift_max cannot be read from this table at all -- it is one row's")
    print("value and its baseline spread is 9x. Use scripts/rank_test.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else r"<runs-B>"))
