# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Serial overnight campaign for the self-calibrating dead zone.
#
# Serial by design: one LichtFeld Studio process at a time. The script owns
# the loop so nothing has to poll files from another process, and a failed
# run is recorded and skipped rather than ending the night. Runs that
# already produced a stats.json are skipped, so re-invoking resumes.

param(
    [string]$Exe     = 'D:\Apps\LichtFeld-Studio\build\Release\LichtFeld-Studio.exe',
    [string]$Data    = '<dataset-B>\colmap',
    [string]$RunRoot = '<runs-B>',
    [string]$Plugin  = '<plugin-repo>',
    [int]$Iters      = 30000,
    [int]$P5Cap      = 16000000,
    [ValidateSet('main', 'followup', 'tailstats', 'newq', 'visible', 'datac', 'dataccrop', 'datacbox', 'datacfull',
                 'datad', 'datadcrop', 'datacq')]
    [string]$Plan    = 'main',
    [string[]]$Only  = @()
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'

# Ordered so the highest-value results land first: if the night dies early,
# P1/P2 still answer the central question (does the control-row estimate
# match a real baseline, and does calibration move the drift tail).
$runs = @(
    @{ id = 'P0_smoke';       en = 1; cal = 1; sp = 2.0; cap = 8000000; it = 2000  ; q = 70 }

    @{ id = 'P1_base_1';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'P2_calib_1';     en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 70 }
    @{ id = 'P1_base_2';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'P2_calib_2';     en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 70 }

    @{ id = 'P3_fixed10_1';   en = 1; cal = 0; sp = 1.0; cap = 8000000; it = $Iters }
    @{ id = 'P3_fixed10_2';   en = 1; cal = 0; sp = 1.0; cap = 8000000; it = $Iters }

    @{ id = 'P4_base_3';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'P4_base_4';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'P4_base_5';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'P4_base_6';      en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }

    @{ id = 'P5_base_hi_1';   en = 0; cal = 0; sp = 2.0; cap = $P5Cap;  it = $Iters }
    @{ id = 'P5_calib_hi_1';  en = 1; cal = 1; sp = 2.0; cap = $P5Cap;  it = $Iters; q = 70 }
    @{ id = 'P5_base_hi_2';   en = 0; cal = 0; sp = 2.0; cap = $P5Cap;  it = $Iters }
    @{ id = 'P5_calib_hi_2';  en = 1; cal = 1; sp = 2.0; cap = $P5Cap;  it = $Iters; q = 70 }
)

# Follow-up plan. Every arm here is compared against the SAME base_8M
# baselines the main plan already produced (n=6), so no baseline is re-run.
#
# q: the main plan showed q70 lands on 0.003306, which is 1.01x the value a
#    manual sweep found, while the inert default (spacing 2.0 = 0.006572)
#    sits at roughly q95. The open direction is therefore TIGHTER -- q60 and
#    q50 pick 0.86x and 0.74x of the manual value. The question they answer
#    is where photometric cost finally appears, since neither q70 nor the
#    manual value showed any.
# start: the default 1000 makes the zone clamp hard and then relax
#    (0.00142 -> 0.00321 over 10 updates). 5000 waits for a more mature
#    distribution. Whether the early clamp helps or hurts is unseparated.
# frac: the control rows over-report free drift by +3.4% (p50) to +7.7%
#    (p95) because they absorb residual the anchored rows no longer take.
#    If that is the mechanism, raising the held-out fraction to 10% spreads
#    the same residual over 5x more free rows and the bias should shrink.
$followup = @(
    @{ id = 'F_q60_1';    en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 60 }
    @{ id = 'F_q60_2';    en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 60 }
    @{ id = 'F_q50_1';    en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 50 }
    @{ id = 'F_q50_2';    en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 50 }

    @{ id = 'F_start5k_1'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; start = 5000; q = 70 }
    @{ id = 'F_start5k_2'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; start = 5000; q = 70 }

    @{ id = 'F_frac10_1'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; frac = 0.10; q = 70 }
    @{ id = 'F_frac10_2'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; frac = 0.10; q = 70 }
)

# Baselines re-run on the build that emits the bounded tail statistics
# (p99/p999 and the escape counts). The 15 main-campaign runs predate them,
# so without these there is nothing to compare a calibrated run's
# escaped_2x = 25 against, and the statistic that replaces the unusable
# `max` cannot be shown to work.
$tailstats = @(
    @{ id = 'T_base_1'; en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'T_base_2'; en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
)

# The two questions that only became visible once the first campaign was in.
#
# G_hi_diag: at 16M cap two identical calibrated runs ended at max drift
#   16.657 and 1.759 with the same relocations, exclusion rate and p95 --
#   one row. These carry the tail diagnostic, which reports whether the
#   extreme drifters sit below min_pull_opacity (never pulled by design)
#   or still carry their original anchor (a relocation the detector missed).
# G_split_q60: scale_p90 inflation grows monotonically as the leash tightens
#   (+2.1/+2.9/+3.4% at q70/q60/q50). DESIGN.md blames MRNF's long-axis
#   split, whose displacement is far below the teleport threshold. These
#   re-anchor split rows and are compared against the existing q60 arm.
#   Interleaved so an early stop still leaves one of each.
$newq = @(
    @{ id = 'G_split_q60_1'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 60; split = 1 }
    @{ id = 'G_hi_diag_1';   en = 1; cal = 1; sp = 2.0; cap = $P5Cap;  it = $Iters; q = 70 }
    @{ id = 'G_split_q60_2'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 60; split = 1 }
    @{ id = 'G_hi_diag_2';   en = 1; cal = 1; sp = 2.0; cap = $P5Cap;  it = $Iters; q = 70 }
)

if ($Plan -eq 'followup')  { $runs = $followup }
if ($Plan -eq 'tailstats') { $runs = $tailstats }
# Exercise the visible-row tail variant: one unanchored baseline and one
# calibrated run, so the new block has both sides of a comparison.
$visible = @(
    @{ id = 'V_base_1';  en = 0; cal = 0; sp = 2.0; cap = 8000000; it = $Iters }
    @{ id = 'V_calib_1'; en = 1; cal = 1; sp = 2.0; cap = 8000000; it = $Iters; q = 60 }
)

# Third dataset (C: 1,111 images, 4,302,501 surveyed points), and the
# first test of nn mode. Two things are being asked at once:
#
#  1. Do the results from the B capture reproduce on a third dataset?
#     The q default and the scale-inflation question both need one.
#  2. Does nn + anchor_new hold the rows density control CREATES? On this
#     capture the operator's own cleanup deleted appended rows at 4.3x the
#     rate of surveyed ones, and the deleted ones sat a median 5.16 point
#     spacings off the surface. index mode never touches them.
#
# nnfree isolates the two halves of nn mode: retargeting the initial rows to
# the nearest surveyed point, versus additionally pulling the grown ones.
# Without it a difference could not be attributed to either.
#
# max_cap 5000000 matches the run the operator actually cleaned, so the
# cleanup-cost metric is comparable to their delivered model. It is also
# saturated (4.3M surveyed points against a 5M cap, full by iter 1000), so
# the 9M arms repeat the comparison with density control actually running.
$datac = @(
    @{ id = 'M_base_1';    en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'M_nn_1';      en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_index_1';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 70 }
    @{ id = 'M_base_2';    en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'M_nn_2';      en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_index_2';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 70 }

    @{ id = 'M_nnfree_1';  en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; q = 70 }
    @{ id = 'M_nnfree_2';  en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; q = 70 }

    @{ id = 'M_base_hi_1'; en = 0; cal = 0; sp = 2.0; cap = 9000000; it = $Iters }
    @{ id = 'M_nn_hi_1';   en = 1; cal = 1; sp = 2.0; cap = 9000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_index_hi_1';en = 1; cal = 1; sp = 2.0; cap = 9000000; it = $Iters; q = 70 }
    @{ id = 'M_base_hi_2'; en = 0; cal = 0; sp = 2.0; cap = 9000000; it = $Iters }
    @{ id = 'M_nn_hi_2';   en = 1; cal = 1; sp = 2.0; cap = 9000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_index_hi_2';en = 1; cal = 1; sp = 2.0; cap = 9000000; it = $Iters; q = 70 }
)

if ($Plan -eq 'newq')      { $runs = $newq }
if ($Plan -eq 'visible')   { $runs = $visible }
# Same comparison on a point cloud cropped to the delivered box (+5% pad).
# The uncropped arms have a floor they cannot beat: 29,175 of the 4,302,501
# surveyed points already lie outside the box, and anchoring HOLDS rows on
# those points there. Cropping the input is what removes that floor -- the
# pad leaves 11,318 of them, which is the realistic case since the delivered
# box is not known before the model is delivered.
# Invoke with -Data ...\colmap_cropped.
$dataccrop = @(
    @{ id = 'M_crop_base_1';  en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'M_crop_nn_1';    en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_crop_index_1'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 70 }
    @{ id = 'M_crop_base_2';  en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'M_crop_nn_2';    en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'M_crop_index_2'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 70 }
)

if ($Plan -eq 'datac')      { $runs = $datac }
# The training-side crop box. 74.8% of the operator's manual removals lay
# outside the delivered model's box, and the engine already de-weights
# splats a crop box rejects (Adam step x0.1, pixel loss x0.1) -- but the box
# is a scene node, not a CLI option, so the plugin sets it. Paired with a
# baseline so the box's own effect is separable from anchoring's.
$datacbox = @(
    @{ id = 'M_box_base_1';  en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters; box = 1 }
    @{ id = 'M_box_nn_1';    en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; box = 1; q = 70 }
    @{ id = 'M_box_base_2';  en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters; box = 1 }
    @{ id = 'M_box_nn_2';    en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; box = 1; q = 70 }
)

if ($Plan -eq 'dataccrop')  { $runs = $dataccrop }
# The whole stack: cloud cropped to the ROI before training, the training
# crop box in force, and nn anchoring holding the grown rows to the surveyed
# surface. Paired with the same stack minus anchoring so the leash's share
# stays separable. Invoke with -Data ...\colmap_cropped.
$datacfull = @(
    @{ id = 'M_full_base_1'; en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters; box = 1 }
    @{ id = 'M_full_nn_1';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; box = 1; q = 70 }
    @{ id = 'M_full_base_2'; en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters; box = 1 }
    @{ id = 'M_full_nn_2';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; box = 1; q = 70 }
)

if ($Plan -eq 'datacbox')   { $runs = $datacbox }
if ($Plan -eq 'datacfull')  { $runs = $datacfull }

# ---------------------------------------------------------------------------
# dataset D: a SECOND vehicle with the operator's own cleanup as the
# label (splat_109800.ply raw vs splat_109800b.ply cleaned). This is the
# replication of the C result on a different vehicle, which is the only
# thing that can promote "crop the input cloud" from one measurement to a
# rule.
#
# The two captures are not the same regime, so the replication is not a
# formality:
#
#                                        C        D
#   operator removed                     1.53%       8.63%   (431,533 splats)
#   of those, outside the box            74.8%       80.3%
#   in-box removal base rate             0.39%       1.823%
#   surveyed points outside the box      0.68%      10.612%
#
# max_cap 5000000 matches the run the operator actually cleaned, so
# cleanup_cost.py is comparable against their delivered model.
#
# Baselines first in each plan: if the night dies early, uncropped base x2
# against cropped base x2 still answers the central question.
$datad = @(
    @{ id = 'L_base_1';    en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'L_base_2';    en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'L_nn_1';      en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'L_nn_2';      en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
)

# The crop uses the NOMINAL box, not this vehicle's delivered box. Cropping to
# the delivered box would leak the label into the input and inflate the
# result; in production the delivered box does not exist yet. The nominal box
# is -1,-1.5,-1:3,0.5,1 (centre [1,-0.5,0], size [4,2,2]) plus the default 5%
# pad, and it contains BOTH vehicles' delivered boxes -- the low-corner gaps
# on this one are [0.001, 0.041, 0.000] and the z faces match exactly.
# Invoke with -Data ...\colmap_cropped (crop_input.py builds it).
$datadcrop = @(
    # Smoke first, and on the CROPPED data deliberately: 2,000 iters verify in
    # minutes that the junctioned dataset loads (rewritten point count,
    # hardlinked images.txt) AND that the plugin fires (applied_iters > 0),
    # before either plan burns hours. A full campaign was once run with both
    # arms silently disabled. Skipped automatically on re-invocation once its
    # stats.json exists.
    @{ id = 'L0_smoke';      en = 1; cal = 1; sp = 2.0; cap = 5000000; it = 2000; nn = 1; anew = 1; q = 70 }

    @{ id = 'L_crop_base_1'; en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'L_crop_base_2'; en = 0; cal = 0; sp = 2.0; cap = 5000000; it = $Iters }
    @{ id = 'L_crop_nn_1';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
    @{ id = 'L_crop_nn_2';   en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; nn = 1; anew = 1; q = 70 }
)

if ($Plan -eq 'datad')      { $runs = $datad }
if ($Plan -eq 'datadcrop')  { $runs = $datadcrop }

# The q sweep on C, closing the HANDOFF item "q の既定や スケール膨張の
# データセット依存性はこのデータでは検証していない". Dataset B found q70
# lands 1.01x from the manually-found optimum, tighter is monotonically
# better for position (p95 -44.9 -> -57.9%) with the cost appearing in
# SHAPE (scale_p90 +2.1 -> +3.4%), not photometry. Whether that transfers
# to a second capture with 4x the points decides if q's default is a
# property of the mechanism or of dataset B.
#
# index mode + calibration, matching M_index_1/2 which ARE the q70 arm.
# Baselines M_base_1/2 already exist -- do not re-run them.
# Invoke with -Data ...\dataset C\colmap -RunRoot ...\lfs_runs_datac.
$datacq = @(
    @{ id = 'M_q60_1'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 60 }
    @{ id = 'M_q50_1'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 50 }
    @{ id = 'M_q60_2'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 60 }
    @{ id = 'M_q50_2'; en = 1; cal = 1; sp = 2.0; cap = 5000000; it = $Iters; q = 50 }
)

if ($Plan -eq 'datacq')     { $runs = $datacq }

# @() so a single surviving run stays an array: bare Where-Object unwraps it
# to one hashtable, whose .Count is its KEY count -- the L0_smoke invocation
# logged "campaign start: 8 run(s)" for one run of 8 keys. Cosmetic (foreach
# still ran the one run), but a wrong run count in a campaign log is exactly
# the line someone checks at 2am.
if ($Only.Count -gt 0) { $runs = @($runs | Where-Object { $Only -contains $_.id }) }

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$log = Join-Path $RunRoot 'campaign_log.txt'

# Logging must not be able to fail silently. A `tail -F` from another shell
# holds this file open on Windows, and Add-Content then throws on every
# call; with $ErrorActionPreference = 'Continue' that produced a campaign
# which ran perfectly for 13 hours while writing nothing after the first
# two entries. Fall back to a sibling file and say so on stdout, which is
# captured even when the log is not.
$script:logBroken = $false
function Write-Log([string]$m) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m)
    Write-Output $line
    try {
        Add-Content -Path $log -Value $line -Encoding utf8 -ErrorAction Stop
    } catch {
        if (-not $script:logBroken) {
            $script:logBroken = $true
            Write-Output ("WARNING: cannot write {0} ({1}). Falling back to {0}.alt -- check for a tail/grep holding it open." -f $log, $_.Exception.Message)
        }
        try {
            Add-Content -Path ($log + '.alt') -Value $line -Encoding utf8 -ErrorAction Stop
        } catch {
            # stdout is the last resort and already has the line.
        }
    }
}

# VRAM/RAM watchdog for the whole campaign. nvidia-smi does the sampling, so
# nothing needs to poll while a training run holds the GPU.
$vramCsv = Join-Path $RunRoot 'vram.csv'
$smi = $null
try {
    $smi = Start-Process -FilePath 'nvidia-smi' `
        -ArgumentList '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 30' `
        -RedirectStandardOutput $vramCsv -NoNewWindow -PassThru
} catch {
    Write-Log ('WARN could not start the VRAM watchdog: ' + $_.Exception.Message)
}

Write-Log ('campaign start: {0} run(s), iters={1}, P5 cap={2}' -f $runs.Count, $Iters, $P5Cap)
Write-Log ('engine: ' + ((& $Exe --version 2>$null) | Select-Object -Last 1))

foreach ($r in $runs) {
    $out = Join-Path $RunRoot $r.id
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $stats = Join-Path $out 'stats.json'

    if (Test-Path $stats) { Write-Log ('SKIP  {0} (stats.json exists)' -f $r.id); continue }

    $env:LFS_MPC_ENABLED             = "$($r.en)"
    $env:LFS_MPC_CALIBRATE           = "$($r.cal)"
    $env:LFS_MPC_FREE_RADIUS_SPACING = "$($r.sp)"
    $env:LFS_MPC_STATS_OUT           = $stats
    $env:LFS_MPC_SNAPSHOT_EVERY      = '1000'
    $env:LFS_MPC_CALIB_EVERY         = '500'
    # Per-run overrides; anything not set falls back to the plugin default.
    # Fallback mirrors the plugin default (60 since 2026-07-31; was 70).
    # Arms that mean q70 -- e.g. datacq's explicit q70 reference arm -- say so.
    $q     = if ($r.ContainsKey('q'))     { $r.q }     else { 60 }
    $start = if ($r.ContainsKey('start')) { $r.start } else { 1000 }
    $frac  = if ($r.ContainsKey('frac'))  { $r.frac }  else { 0.02 }
    $split = if ($r.ContainsKey('split')) { $r.split } else { 0 }
    $env:LFS_MPC_REANCHOR_SPLIT = "$split"
    $nn    = if ($r.ContainsKey('nn'))    { $r.nn }    else { 0 }
    $anew  = if ($r.ContainsKey('anew'))  { $r.anew }  else { 0 }
    $env:LFS_MPC_MODE       = $(if ($nn -eq 1) { 'nn' } else { 'index' })
    $env:LFS_MPC_ANCHOR_NEW = "$anew"
    $env:LFS_MPC_NN_REFRESH = '100'
    # Nominal ROI for this vehicle class, from the delivered model's own box.
    $box = if ($r.ContainsKey('box')) { $r.box } else { 0 }
    $env:LFS_MPC_CROP_BOX = $(if ($box -eq 1) { '-0.8777,-1.2579,-1.0:3.1204,0.3682,1.0' } else { '' })
    $env:LFS_MPC_CROP_BOX_PAD = '0.05'
    $env:LFS_MPC_CALIB_Q          = "$q"
    $env:LFS_MPC_CALIB_START      = "$start"
    $env:LFS_MPC_CONTROL_FRACTION = "$frac"

    Write-Log ('START {0}: enabled={1} calibrate={2} spacing={3} max_cap={4} iters={5} q={6} start={7} frac={8} split={9} mode={10} anchor_new={11}' `
        -f $r.id, $r.en, $r.cal, $r.sp, $r.cap, $r.it, $q, $start, $frac, $split, $env:LFS_MPC_MODE, $anew)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & $Exe --headless --train -d $Data -o $out --iter $r.it `
        --max-cap $r.cap --strategy mrnf --eval --test-every 8 `
        --python-script (Join-Path $Plugin 'headless_anchor.py') `
        --log-file (Join-Path $out 'lfs.log') --log-level info | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()

    if ($code -ne 0) {
        Write-Log ('FAIL  {0}: exit={1} after {2:N1} min -- continuing' `
            -f $r.id, $code, $sw.Elapsed.TotalMinutes)
        continue
    }
    if (-not (Test-Path $stats)) {
        Write-Log ('WARN  {0}: exit 0 but no stats.json' -f $r.id)
        continue
    }
    $j = Get-Content $stats -Raw | ConvertFrom-Json
    Write-Log ('DONE  {0}: {1:N1} min, applied_iters={2}, free_radius={3}' `
        -f $r.id, $sw.Elapsed.TotalMinutes, $j.applied_iters, $j.config.free_radius_effective)
}

if ($smi -and -not $smi.HasExited) { Stop-Process -Id $smi.Id -Force }
Write-Log 'campaign complete'
