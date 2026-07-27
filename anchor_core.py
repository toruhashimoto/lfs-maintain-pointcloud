# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core position-anchor regularizer for LichtFeld Studio.

Keeps Gaussian means close to their initial (pre-placed) point-cloud
positions during training without hard-freezing them.

Mechanism
---------
LichtFeld Studio's Python hooks all execute in the SafeControl window
*after* the full train step (forward + backward + fused Adam + MCMC
densification) of each iteration, and the optimizer gradient buffers are
not exposed to Python. A pre-Adam gradient injection is therefore
impossible from a plugin; instead we apply the mathematically equivalent
*proximal correction* directly on the live means tensor (a writable CUDA
view into the optimizer's own parameter memory):

    d_i      = mu_i - a_i                       # displacement from anchor
    r_i      = ||d_i||
    excess_i = max(0, r_i - free_radius)        # dead zone
    pull_i   = s(t) * min(excess_i, huber_delta)  # bounded influence
    mu_i    -= (d_i / r_i) * pull_i * gate_i

This is one explicit (sub)gradient step per iteration on

    L_anchor = sum_i gate_i * rho_delta( max(0, ||mu_i - a_i|| - r_free) )

i.e. a Huber penalty with a dead zone, exactly the soft position anchor
used by the LI-GS / GeomGS / EnerGS family, minus Adam's moment
accumulation (the correction is deterministic and scale-controlled).

With strength = 1 and huber_delta = 0 (disabled) the correction becomes a
hard projection onto the free-radius ball around each anchor (a "leash");
with small strength it is a soft spring. strength = 0 disables the pull
while topology tracking continues.

MCMC topology handling (mode = "index")
---------------------------------------
MCMC never permutes rows: relocation clones an alive row into a dead
row's slot in place, growth appends rows at the tail, pruning is a soft
delete. We therefore keep a per-row anchor buffer and:

  * appended rows      -> anchored at their birth position
                          (or left unanchored if anchor_new_splats=False)
  * relocated rows     -> detected as a large positional jump between
                          consecutive iterations on a row that is alive,
                          or as a dead->alive opacity discontinuity
                          (relocation copies a sampled alive opacity,
                          a jump Adam cannot produce in one step);
                          re-anchored at the relocation target
  * near-dead rows     -> receive no pull at all (min_pull_opacity):
                          MCMC noise moves them freely, they are invisible,
                          and their anchors are irrelevant until relocation
  * soft-deleted rows  -> harmless, pull on them is irrelevant

If an implausibly large fraction of rows (> 5%) teleports in a single
iteration, it is treated as a whole-model event (e.g. Bake Transform,
undo of a bake, external means write) and anchors are re-captured with a
warning instead of per-row re-anchoring.

mode = "nn" additionally re-targets each row's anchor to the nearest
point of a fixed reference cloud (the immutable initial snapshot or an
external PLY) every nn_refresh iterations. This survives any topology
change by construction and can anchor late-born splats to the reference
surface, at the cost of a periodic CPU nearest-neighbor query (scipy).

Thread safety: all state mutation happens on the training thread inside
apply(). The UI panel only writes scalar parameters and request flags
(GIL-atomic); use request_reset() / request_capture() from other threads
instead of reset() / capture_now().
"""

import json
import os

import lichtfeld as lf

try:  # package import (plugin) vs. flat import (--python-script)
    from .stats_policy import should_snapshot
except ImportError:  # pragma: no cover - exercised by headless_anchor.py
    from stats_policy import should_snapshot

_EPS = 1e-12
_BIG = 1e30

# Relocation-detection constants (see MCMC notes in the module docstring).
_RELOC_MIN_OPACITY = 0.02   # positional jumps only count for rows this alive
_REVIVE_FROM = 0.02         # dead->alive opacity discontinuity: below this...
_REVIVE_TO = 0.10           # ...to above this in one step = relocation
_MASS_TELEPORT_FRACTION = 0.05  # more than this fraction -> whole-model event


def apply_env_overrides(reg):
    """Apply LFS_MPC_* environment variables onto a regularizer.

    Lets headless/CI runs configure the plugin without the GUI panel.
    Only variables that are actually set override the current values.
    """
    def _f(name, cur):
        v = os.environ.get(name)
        try:
            return float(v) if v is not None else cur
        except ValueError:
            return cur

    def _i(name, cur):
        v = os.environ.get(name)
        try:
            return int(v) if v is not None else cur
        except ValueError:
            return cur

    def _b(name, cur):
        v = os.environ.get(name)
        if v is None:
            return cur
        return v.strip().lower() in ("1", "true", "yes", "on")

    cfg_path = os.environ.get("LFS_MPC_CONFIG", "")
    if cfg_path and os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                reg.load_config(json.load(f))
        except (OSError, ValueError) as e:
            lf.log.error(f"[maintain_pointcloud] bad LFS_MPC_CONFIG: {e}")
    reg.enabled = _b("LFS_MPC_ENABLED", reg.enabled)
    reg.strength = _f("LFS_MPC_STRENGTH", reg.strength)
    reg.free_radius = _f("LFS_MPC_FREE_RADIUS", reg.free_radius)
    reg.free_radius_spacing = _f("LFS_MPC_FREE_RADIUS_SPACING",
                                 reg.free_radius_spacing)
    reg.huber_delta = _f("LFS_MPC_HUBER_DELTA", reg.huber_delta)
    reg.max_distance = _f("LFS_MPC_MAX_DISTANCE", reg.max_distance)
    reg.opacity_gate = _b("LFS_MPC_OPACITY_GATE", reg.opacity_gate)
    reg.min_pull_opacity = _f("LFS_MPC_MIN_PULL_OPACITY", reg.min_pull_opacity)
    reg.anchor_new_splats = _b("LFS_MPC_ANCHOR_NEW", reg.anchor_new_splats)
    reg.warmup_iters = _i("LFS_MPC_WARMUP", reg.warmup_iters)
    reg.start_iter = _i("LFS_MPC_START", reg.start_iter)
    reg.stop_iter = _i("LFS_MPC_STOP", reg.stop_iter)
    reg.teleport_threshold = _f("LFS_MPC_TELEPORT", reg.teleport_threshold)
    reg.mode = os.environ.get("LFS_MPC_MODE", reg.mode)
    reg.reference_ply = os.environ.get("LFS_MPC_REF_PLY", reg.reference_ply)
    reg.nn_refresh = _i("LFS_MPC_NN_REFRESH", reg.nn_refresh)
    reg.log_every = _i("LFS_MPC_LOG_EVERY", reg.log_every)
    reg.stats_out = os.environ.get("LFS_MPC_STATS_OUT", reg.stats_out)
    reg.stats_snapshot_every = _i("LFS_MPC_SNAPSHOT_EVERY",
                                  reg.stats_snapshot_every)
    return reg


class AnchorRegularizer:
    """Soft position anchor applied in the per-iteration hook window."""

    def __init__(self):
        # --- user parameters -------------------------------------------------
        self.enabled = False
        # Pull strength per iteration, 0..1. Fraction of the excess
        # displacement removed each step. 1.0 + huber_delta==0 => hard leash.
        # NOTE on the default: a constant strength cannot stay "soft". The
        # trainer's effective means learning rate varies ~38x over a run
        # (MRNF: means_lr * decay^t * bounds.median_size(t)), while the
        # equilibrium displacement of this proximal scheme is
        # (per-iteration photometric push) / strength -- so a fixed strength
        # tightens monotonically and becomes a hard freeze in the final third
        # of every run. Measured at strength=0.10 on a real 30k run: init-row
        # drift median 0.0837 -> 7.8e-05, i.e. frozen, with splat max-axis
        # scale inflating 2.2x in compensation. The dead zone below is what
        # makes the anchor structurally soft; strength then only controls how
        # fast a runaway is reeled back to the edge of that zone.
        self.strength = 0.01
        # Dead zone radius (world units). Inside it there is zero pull, so
        # splats can settle onto local surface detail. 0 = derive it at
        # capture time from the point cloud itself (see free_radius_spacing).
        self.free_radius = 0.0
        # Auto dead zone = this multiple of the captured cloud's median
        # nearest-neighbour spacing, used when free_radius == 0. Splats need
        # roughly 1-2 local point spacings to settle onto surface detail
        # (measured: free drift is 1.98x the local spacing at the median).
        # Set to 0 to disable auto-derivation and get a pure spring.
        self.free_radius_spacing = 2.0
        # Huber clamp: per-iteration pull magnitude never exceeds
        # strength * huber_delta (bounded influence). 0 = disabled.
        # A value > 0 also bounds the damage of any undetected stale anchor.
        self.huber_delta = 0.0
        # Splats farther than this from their anchor get no pull at all
        # (treated as legitimately detached geometry). 0 = infinite.
        self.max_distance = 0.0
        # Multiply the pull by sigmoid(opacity_raw).
        self.opacity_gate = False
        # Rows with sigmoid(opacity_raw) below this get no pull at all:
        # they are invisible, MCMC noise moves them freely, and pulling
        # them only fights the relocation machinery. 0 = disabled.
        self.min_pull_opacity = 0.01
        # Linear ramp of strength over the first N iterations after start.
        self.warmup_iters = 300
        self.start_iter = 0
        self.stop_iter = 0  # 0 = never stop
        # Anchor rows appended/relocated by MCMC at their birth position.
        # Default False: the plugin's purpose is to hold the PRE-PLACED
        # cloud, and an MCMC birth/relocation position carries no geometric
        # meaning. Measured with this set to True on a real 30k run, only
        # 219198/5000000 = 4.4% of the final model was still anchored to an
        # init point -- 95.6% of the pull acted on arbitrary sample positions,
        # which pinned freshly relocated splats to their landing sites
        # (relocations +48%) and let the grown population spread instead of
        # migrating onto surfaces. Set True only if you want "freeze the
        # model wherever each splat was born", which is a different goal.
        self.anchor_new_splats = False
        # Row jump between consecutive iterations larger than this counts
        # as an MCMC relocation (teleport). 0 = auto (1% of scene diagonal).
        self.teleport_threshold = 0.0
        # "index": per-row birth anchors. "nn": nearest neighbor of a fixed
        # reference cloud, refreshed every nn_refresh iterations.
        self.mode = "index"
        self.reference_ply = ""  # nn mode: external reference cloud (optional)
        self.nn_refresh = 100
        # Stats cadence (GPU->CPU scalar syncs); logging cadence.
        self.stats_every = 10
        self.log_every = 0  # 0 = no periodic log
        # Where to write the drift-stats JSON ("" = LFS_MPC_STATS_OUT, if set).
        self.stats_out = ""
        # How often to re-write that JSON from the post_step hook. This is not
        # a convenience: LichtFeld Studio v0.5.1 registers the training_end
        # hook but never dispatches it in headless mode, and its embedded
        # interpreter does not run atexit handlers either, so a run that only
        # wrote stats at the end would produce no file at all. Snapshotting
        # makes the last successful write the final result. 0 disables it.
        self.stats_snapshot_every = 1000

        # --- cross-thread request flags (set by UI, serviced in apply()) -----
        self._reset_requested = False
        self._capture_requested = False

        # --- state ------------------------------------------------------------
        self._anchor0 = None      # immutable copy of the initial capture
        self._anchor = None       # [N,3] lf.Tensor (cuda)
        self._mask = None         # [N,1] lf.Tensor, 1.0 = anchored
        self._orig = None         # [N,1] lf.Tensor, 1.0 = still the original
        #                           init point captured in this slot (stats)
        self._prev = None         # [N,3] means at end of previous hook
        self._prev_op = None      # [N,1] opacity_raw at end of previous hook
        self._n = 0
        self._n0 = 0              # row count at capture (init cloud size)
        self._scene_diag = 0.0
        self._auto_free_radius = 0.0  # derived from cloud spacing at capture
        self._nn_spacing = 0.0
        self._baseline_valid = True   # False after any mid-run re-capture
        self._recaptured_at = -1
        self._ref_np = None       # nn mode: reference cloud as numpy [M,3]
        self._kdtree = None
        self._ref_built_from = None   # reference_ply value the tree was built
        #                               from (recorded even on failed loads)
        self._last_nn_iter = -1
        self._last_snapshot_it = -1   # iteration of the last stats file write

        # --- read-only stats (panel display) ----------------------------------
        self.stat_iter = 0
        self.stat_mean_excess = 0.0
        self.stat_max_excess = 0.0
        self.stat_anchored_rows = 0
        self.stat_rows = 0
        self.stat_teleports = 0
        self.stat_appended = 0
        self.stat_applied_iters = 0
        self.stat_errors = 0
        self.stat_captured = False

    # ------------------------------------------------------------------ utils
    def reset(self):
        """Drop all state; anchors re-capture on the next hook call.

        Call only from the training thread (or while training is stopped);
        from the UI use request_reset().
        """
        self._anchor0 = None
        self._anchor = None
        self._mask = None
        self._orig = None
        self._prev = None
        self._prev_op = None
        self._n = 0
        self._n0 = 0
        self._scene_diag = 0.0
        self._auto_free_radius = 0.0
        self._nn_spacing = 0.0
        self._baseline_valid = True
        self._recaptured_at = -1
        self._ref_np = None
        self._kdtree = None
        self._ref_built_from = None
        self._last_nn_iter = -1
        self._last_snapshot_it = -1
        self.stat_teleports = 0
        self.stat_appended = 0
        self.stat_applied_iters = 0
        self.stat_errors = 0
        self.stat_captured = False

    def request_reset(self):
        """Thread-safe: ask the training thread to reset at the next hook."""
        self._reset_requested = True

    def request_capture(self):
        """Thread-safe: ask the training thread to re-capture anchors."""
        self._capture_requested = True

    def _model(self):
        scene = lf.get_scene()
        if scene is None:
            return None
        try:
            return scene.training_model()
        except Exception:
            return None

    def _row_norm(self, t):
        """[N,3] -> [N,1] Euclidean norm."""
        return (t * t).sum(dim=1, keepdim=True).sqrt()

    def _bool_f(self, cond, n):
        """[N,1] bool -> [N,1] float 0/1 (no bitwise ops in lf.Tensor)."""
        return lf.Tensor.where(
            cond,
            lf.Tensor.ones([n, 1], device="cuda", dtype="float32"),
            lf.Tensor.zeros([n, 1], device="cuda", dtype="float32"),
        )

    @staticmethod
    def _median_nn_spacing(pos, sample=200000):
        """Median nearest-neighbour distance of a point cloud (subsampled)."""
        try:
            import numpy as np
            from scipy.spatial import cKDTree
        except ImportError:
            return 0.0
        if len(pos) > sample:
            step = len(pos) // sample
            pos = pos[::step]
        d, _ = cKDTree(pos).query(pos, k=2, workers=-1)
        return float(np.median(d[:, 1]))

    def _effective_free_radius(self):
        """User value if set, else the spacing-derived dead zone."""
        return self.free_radius if self.free_radius > 0.0 else self._auto_free_radius

    def _teleport_thresh(self):
        if self.teleport_threshold > 0.0:
            return self.teleport_threshold
        return 0.01 * self._scene_diag if self._scene_diag > 0.0 else 0.0

    def _strength_at(self, it):
        if not self.enabled:
            return 0.0
        if it < self.start_iter:
            return 0.0
        if self.stop_iter > 0 and it >= self.stop_iter:
            return 0.0
        s = max(0.0, min(1.0, self.strength))
        if self.warmup_iters > 0:
            ramp = (it - self.start_iter) / float(self.warmup_iters)
            s *= max(0.0, min(1.0, ramp))
        return s

    # ---------------------------------------------------------------- capture
    def _capture(self, means, opacity_raw, it=0, initial=True):
        n = means.shape[0]
        self._anchor = means.clone()
        self._anchor0 = means.clone()
        self._prev = means.clone()
        self._prev_op = opacity_raw.clone() if opacity_raw is not None else None
        self._n = n
        self._n0 = n
        self._mask = lf.Tensor.ones([n, 1], device="cuda", dtype="float32")
        self._orig = lf.Tensor.ones([n, 1], device="cuda", dtype="float32")
        if not initial:
            self._baseline_valid = False
            self._recaptured_at = it
        try:
            # Robust extent: a raw min/max bounding box is dominated by the
            # handful of far-flung outlier points every SfM cloud carries
            # (observed: 135 stray points inflating a 37-unit object to a
            # 332000-unit box), which would make the auto teleport threshold
            # thousands of times too large. Use a per-axis 1-99 percentile
            # span instead; on a clean cloud this equals the bounding box.
            import numpy as np
            pos = means.numpy(copy=True).reshape(-1, 3)
            lo = np.percentile(pos, 1.0, axis=0)
            hi = np.percentile(pos, 99.0, axis=0)
            self._scene_diag = float(np.linalg.norm(hi - lo))
            self._nn_spacing = self._median_nn_spacing(pos)
            self._auto_free_radius = self.free_radius_spacing * self._nn_spacing
        except Exception:
            self._scene_diag = 0.0
            self._nn_spacing = 0.0
            self._auto_free_radius = 0.0
            lf.log.warn(
                "[maintain_pointcloud] scene diagonal computation failed; "
                "auto teleport detection is DISABLED (set teleport_threshold "
                "explicitly to re-enable relocation tracking)"
            )
        self.stat_captured = True
        lf.log.info(
            f"[maintain_pointcloud] anchors captured: {n} rows, "
            f"scene diag ~{self._scene_diag:.4f}, "
            f"nn spacing ~{self._nn_spacing:.5f}, "
            f"free radius {self._effective_free_radius():.5f}"
            f"{' (auto)' if self.free_radius <= 0.0 else ''}, "
            f"teleport thresh {self._teleport_thresh():.4f}"
            + ("" if initial else f" (RE-capture at iter {it}; init-drift "
               f"baseline no longer refers to training start)")
        )

    def capture_now(self):
        """Re-capture anchors at current positions.

        Training-thread / stopped-training use only; from the UI panel use
        request_capture() instead (serviced at the next hook).
        """
        m = self._model()
        if m is None:
            return False
        self.reset()
        self._capture(m.means_raw, m.opacity_raw)
        return True

    # ----------------------------------------------------------- topology sync
    def _sync_topology(self, means, opacity_raw, it):
        """Handle MCMC growth (append) and relocation (teleport) in-place."""
        n_now = means.shape[0]

        # Self-heal: if a previous iteration died mid-update (swallowed
        # exception, OOM during a growth append), buffers may disagree.
        # A corrupt state is unrecoverable row-wise -> full re-capture.
        if (self._anchor is None or self._mask is None or self._orig is None
                or self._anchor.shape[0] != self._n
                or self._mask.shape[0] != self._n
                or self._orig.shape[0] != self._n):
            lf.log.warn(
                "[maintain_pointcloud] internal state inconsistent "
                "(interrupted update?); re-capturing anchors"
            )
            self._capture(means, opacity_raw, it, initial=False)
            return

        if n_now < self._n:
            # Unexpected shrink (checkpoint reload, manual edit): re-capture.
            lf.log.warn(
                f"[maintain_pointcloud] row count shrank {self._n}->{n_now}; "
                "re-capturing anchors at current positions"
            )
            self._capture(means, opacity_raw, it, initial=False)
            return

        if n_now > self._n:
            # MCMC growth: appended rows anchor at their birth position.
            new_rows = means[self._n:n_now]
            grown_n = n_now - self._n
            self._anchor = lf.Tensor.cat([self._anchor, new_rows.clone()], dim=0)
            fill = 1.0 if self.anchor_new_splats else 0.0
            self._mask = lf.Tensor.cat(
                [self._mask,
                 lf.Tensor.full([grown_n, 1], fill, device="cuda", dtype="float32")],
                dim=0)
            self._orig = lf.Tensor.cat(
                [self._orig,
                 lf.Tensor.zeros([grown_n, 1], device="cuda", dtype="float32")],
                dim=0)
            if self._prev is not None:
                self._prev = lf.Tensor.cat([self._prev, new_rows.clone()], dim=0)
            if self._prev_op is not None and opacity_raw is not None:
                self._prev_op = lf.Tensor.cat(
                    [self._prev_op, opacity_raw[self._n:n_now].clone()], dim=0)
            self.stat_appended += grown_n
            self._n = n_now

        # Relocation (teleport) detection needs a valid previous snapshot;
        # after an interrupted iteration just refresh and skip one check.
        if (self._prev is None or self._prev.shape[0] != n_now
                or self._prev_op is None or opacity_raw is None
                or self._prev_op.shape[0] != n_now):
            self._prev = means.clone()
            if opacity_raw is not None:
                self._prev_op = opacity_raw.clone()
            return

        thresh = self._teleport_thresh()
        if thresh <= 0.0:
            return

        # An MCMC relocation copies an alive row's params into a dead slot:
        # large positional jump AND the row is now alive. Pure noise kicks
        # on near-dead rows also jump, but stay near-transparent -> gate on
        # current opacity. A relocation whose jump is sub-threshold is
        # caught by the dead->alive opacity discontinuity instead (Adam
        # cannot move sigmoid(op) from <0.02 to >0.10 in one step).
        jump = self._row_norm(means - self._prev)              # [N,1]
        cur_sig = opacity_raw.sigmoid()                        # [N,1]
        prev_sig = self._prev_op.sigmoid()                     # [N,1]
        jumped_f = self._bool_f(jump > thresh, n_now)
        alive_f = self._bool_f(cur_sig >= _RELOC_MIN_OPACITY, n_now)
        revive_f = (self._bool_f(prev_sig < _REVIVE_FROM, n_now)
                    * self._bool_f(cur_sig > _REVIVE_TO, n_now))
        tele_f = (jumped_f * alive_f + revive_f).clamp(0.0, 1.0)  # [N,1] 0/1
        n_tele = int(tele_f.sum_scalar())
        if n_tele == 0:
            return

        if n_tele > max(1000.0, _MASS_TELEPORT_FRACTION * n_now):
            # Whole-model event (Bake Transform, undo of a bake, external
            # means write): per-row re-anchoring would silently discard the
            # captured cloud, so re-capture loudly instead.
            lf.log.warn(
                f"[maintain_pointcloud] {n_tele}/{n_now} rows jumped in one "
                "iteration - treating as a whole-model transform (bake/undo/"
                "external edit) and re-capturing anchors"
            )
            self._capture(means, opacity_raw, it, initial=False)
            return

        tele = tele_f > 0.5                                    # [N,1] bool
        tele3 = tele.expand([n_now, 3])
        # Re-anchor relocated rows at their new birth position.
        self._anchor = lf.Tensor.where(tele3, means, self._anchor)
        fill = 1.0 if self.anchor_new_splats else 0.0
        self._mask = lf.Tensor.where(
            tele,
            lf.Tensor.full([n_now, 1], fill, device="cuda", dtype="float32"),
            self._mask)
        # These slots no longer hold their original init point (stats).
        self._orig = lf.Tensor.where(
            tele, lf.Tensor.zeros([n_now, 1], device="cuda", dtype="float32"),
            self._orig)
        self.stat_teleports += n_tele

    # ------------------------------------------------------------------ nn mode
    def _ensure_reference(self):
        """Build (or rebuild) the KD-tree over the fixed reference cloud.

        Rebuilds whenever reference_ply changed since the last build; the
        attempted path is recorded even when the load fails so a bad path
        is not retried every iteration - only when the user edits it.
        """
        if self._kdtree is not None and self.reference_ply == self._ref_built_from:
            return True
        try:
            import numpy as np  # noqa: F401
            from scipy.spatial import cKDTree
        except ImportError:
            lf.log.error(
                "[maintain_pointcloud] nn mode needs numpy+scipy in the "
                "plugin venv; falling back to index mode"
            )
            self.mode = "index"
            return False
        self._kdtree = None
        self._ref_np = None
        if self.reference_ply:
            try:
                pts, _colors = lf.io.load_point_cloud(self.reference_ply)
                self._ref_np = pts.numpy(copy=True).astype("float32").reshape(-1, 3)
            except Exception as e:
                lf.log.error(
                    f"[maintain_pointcloud] failed to load reference PLY "
                    f"'{self.reference_ply}': {e}; using init snapshot"
                )
                self._ref_np = None
        if self._ref_np is None:
            # Immutable initial snapshot - NOT the live per-row anchor
            # buffer, which accumulates MCMC relocation targets over time.
            self._ref_np = self._anchor0.numpy(copy=True).reshape(-1, 3)
        self._kdtree = cKDTree(self._ref_np)
        self._ref_built_from = self.reference_ply
        self._last_nn_iter = -1  # apply the new reference at the next hook
        src = self.reference_ply if self.reference_ply else "init snapshot"
        lf.log.info(
            f"[maintain_pointcloud] nn reference cloud rebuilt from {src}: "
            f"{self._ref_np.shape[0]} points"
        )
        return True

    def _nn_retarget(self, means, it):
        """Re-target every row's anchor to its nearest reference point."""
        if not self._ensure_reference():
            return
        if self._last_nn_iter >= 0 and it - self._last_nn_iter < max(1, self.nn_refresh):
            return
        mu = means.numpy(copy=True).reshape(-1, 3)
        _dist, idx = self._kdtree.query(mu, workers=-1)
        target = self._ref_np[idx]  # [N,3] float32
        self._anchor = lf.Tensor.from_numpy(target).cuda()
        self._last_nn_iter = it

    # ------------------------------------------------------------------- apply
    def apply(self, hook):
        """Per-iteration hook body. `hook` is the payload dict (or None)."""
        it = 0
        if isinstance(hook, dict):
            it = int(hook.get("iter", hook.get("iteration", 0)))
        self.stat_iter = it

        m = self._model()
        if m is None:
            if self._anchor is not None:
                self.reset()
            return
        means = m.means_raw
        opacity_raw = m.opacity_raw
        n = means.shape[0]
        if n == 0:
            return

        # Service UI-thread requests here so ALL state mutation happens on
        # the training thread (panel buttons only set boolean flags).
        if self._reset_requested:
            self._reset_requested = False
            self._capture_requested = False
            self.reset()
            return
        if self._capture_requested:
            self._capture_requested = False
            self.reset()
            self._capture(means, opacity_raw, it, initial=False)
            return

        if self._anchor is None:
            self._capture(means, opacity_raw, it)
            return

        # Snapshot panel-tunable scalars once (the UI thread may edit them
        # between any two statements below).
        strength_s = self._strength_at(it)
        free_radius = self._effective_free_radius()
        huber_delta = self.huber_delta
        max_distance = self.max_distance
        min_pull_op = self.min_pull_opacity
        opacity_gate = self.opacity_gate
        stats_every = self.stats_every
        log_every = self.log_every

        try:
            self._sync_topology(means, opacity_raw, it)
            if self.mode == "nn":
                self._nn_retarget(means, it)

            # Before the zero-strength early return below: a baseline run
            # (enabled=False) still has to produce drift statistics, since
            # that is the noise floor every effect size is measured against.
            self._maybe_snapshot_stats(it)

            want_stats = stats_every > 0 and (it % stats_every == 0)
            if strength_s <= 0.0 and not want_stats:
                self._prev = means.clone()
                self._prev_op = opacity_raw.clone()
                return

            n = means.shape[0]
            d = means - self._anchor                    # [N,3]
            r = self._row_norm(d)                       # [N,1]
            excess = (r - free_radius).relu() if free_radius > 0.0 else r

            if want_stats:
                gated = excess * self._mask
                anchored = float(self._mask.sum_scalar())
                self.stat_mean_excess = (float(gated.sum_scalar()) / anchored
                                         if anchored > 0.0 else 0.0)
                self.stat_max_excess = float(gated.max_scalar())
                self.stat_rows = n
                self.stat_anchored_rows = int(anchored)

            if strength_s > 0.0:
                hi = huber_delta if huber_delta > 0.0 else _BIG
                pull = excess.clamp(0.0, hi) * strength_s   # [N,1]
                pull = pull * self._mask
                if max_distance > 0.0:
                    pull = lf.Tensor.where(
                        r <= max_distance, pull, lf.Tensor.zeros_like(pull))
                sig = None
                if min_pull_op > 0.0 or opacity_gate:
                    sig = opacity_raw.sigmoid()
                if min_pull_op > 0.0:
                    # Invisible rows: no pull (noise moves them freely and
                    # their anchors are irrelevant until relocation).
                    pull = pull * self._bool_f(sig >= min_pull_op, n)
                if opacity_gate:
                    pull = pull * sig
                coef = pull / r.clamp(_EPS, _BIG)           # [N,1]
                means -= d * coef.expand([n, 3])            # in-place update
                self.stat_applied_iters += 1

            self._prev = means.clone()
            self._prev_op = opacity_raw.clone()
        except Exception as e:
            # The hook dispatcher would swallow this anyway; make the state
            # explicitly safe instead: drop the prev snapshots so the next
            # iteration skips teleport detection rather than comparing
            # against a stale frame, and let _sync_topology's shape checks
            # repair any partially-updated buffers.
            self.stat_errors += 1
            self._prev = None
            self._prev_op = None
            lf.log.error(f"[maintain_pointcloud] apply() failed at iter {it}: {e!r}")
            return

        if log_every > 0 and it % log_every == 0:
            lf.log.info(
                f"[maintain_pointcloud] iter={it} s={strength_s:.4f} "
                f"mean_excess={self.stat_mean_excess:.6f} "
                f"max_excess={self.stat_max_excess:.6f} "
                f"anchored={self.stat_anchored_rows}/{self.stat_rows} "
                f"teleports={self.stat_teleports} appended={self.stat_appended}"
            )

    # -------------------------------------------------------------- lifecycle
    def on_training_start(self, _hook):
        # Fresh run: anchors will be captured lazily on the first
        # per-iteration hook (rows 0..N0 are exactly the init cloud there).
        self.reset()

    def _resolve_stats_out(self):
        """Stats path: explicit parameter wins, else the environment."""
        return self.stats_out or os.environ.get("LFS_MPC_STATS_OUT", "")

    def _write_stats(self, out, stats, it, final):
        """Overwrite the stats JSON. Returns True on success.

        `applied_iters` belongs in the file rather than only in the end-of-run
        log line: on runtimes that never dispatch training_end (see
        stats_snapshot_every) that log line is never emitted either, and
        `applied_iters == 0` is the one check that distinguishes "the anchor
        did nothing" from "the anchor did nothing measurable".
        """
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"enabled": self.enabled,
                           "iter": it,
                           "final": final,
                           "applied_iters": self.stat_applied_iters,
                           "teleports": self.stat_teleports,
                           "appended": self.stat_appended,
                           "errors": self.stat_errors,
                           "config": self.config_dict(),
                           "init_drift": stats}, f, indent=2)
            return True
        except OSError as e:
            lf.log.error(f"[maintain_pointcloud] stats write failed: {e}")
            return False

    def _maybe_snapshot_stats(self, it):
        """Periodically write the stats JSON from inside the post_step hook.

        LichtFeld Studio v0.5.1 registers the training_end hook but never
        dispatches it in headless mode, and its embedded interpreter does not
        run atexit handlers, so on_training_end alone yields no file at all.
        Writing here makes the last successful write the final result; the
        `iter`/`final` fields say which iteration it reflects.
        """
        out = self._resolve_stats_out()
        if not out:
            return
        if not should_snapshot(it, self._last_snapshot_it,
                               self.stats_snapshot_every):
            return
        # Measurement must never be able to damage the regularizer: a failure
        # here is contained instead of unwinding into apply()'s error path,
        # which would drop the teleport-detection snapshots for this iteration.
        try:
            stats = self.init_drift_stats()
            if stats is None:
                return
            if self._write_stats(out, stats, it, final=False):
                self._last_snapshot_it = it
        except Exception as e:
            self._last_snapshot_it = it   # do not retry every iteration
            lf.log.error(
                f"[maintain_pointcloud] stats snapshot failed at iter {it}: {e!r}")

    def on_training_end(self, _hook):
        stats = self.init_drift_stats()
        if stats is not None:
            lf.log.info("[maintain_pointcloud] init drift: " + json.dumps(stats))
        lf.log.info(
            f"[maintain_pointcloud] training end: applied {self.stat_applied_iters} "
            f"corrections, teleports={self.stat_teleports}, "
            f"appended={self.stat_appended}, errors={self.stat_errors}, "
            f"final mean_excess={self.stat_mean_excess:.6f}, "
            f"max_excess={self.stat_max_excess:.6f}"
        )
        out = self._resolve_stats_out()
        if out and stats is not None:
            self._write_stats(out, stats, self.stat_iter, final=True)

    # ------------------------------------------------------------- measurement
    def init_drift_stats(self):
        """Displacement of the original init rows from their captured
        positions, EXCLUDING rows whose slot was reused by an MCMC
        relocation (those would report the relocation distance, not drift).

        Returns a dict with mean/median/p95/max drift over the surviving
        original rows, coverage counts, and a baseline_valid flag (False
        if anchors were re-captured mid-run, in which case the baseline is
        the re-capture state rather than training start).
        """
        m = self._model()
        if m is None or self._anchor0 is None or self._n0 == 0:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        n0 = min(self._n0, m.means_raw.shape[0])
        mu = m.means_raw[0:n0].numpy(copy=True).reshape(-1, 3)
        a0 = self._anchor0[0:n0].numpy(copy=True).reshape(-1, 3)
        keep = self._orig[0:n0].numpy(copy=True).reshape(-1) > 0.5
        d_all = np.linalg.norm(mu - a0, axis=1)
        d = d_all[keep]
        if d.size == 0:
            d = d_all  # degenerate: everything relocated; report unfiltered
        return {
            "rows": int(n0),
            "rows_measured": int(keep.sum()),
            "rows_excluded": int(n0 - keep.sum()),
            "baseline_valid": bool(self._baseline_valid),
            "recaptured_at_iter": int(self._recaptured_at),
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
            "scene_diag": float(self._scene_diag),
            "teleports": int(self.stat_teleports),
            "appended": int(self.stat_appended),
            "errors": int(self.stat_errors),
            "applied_iters": int(self.stat_applied_iters),
        }

    # ------------------------------------------------------------ configuration
    def config_dict(self):
        return {
            "enabled": self.enabled,
            "strength": self.strength,
            "free_radius": self.free_radius,
            "free_radius_spacing": self.free_radius_spacing,
            "free_radius_effective": self._effective_free_radius(),
            "nn_spacing": self._nn_spacing,
            "huber_delta": self.huber_delta,
            "max_distance": self.max_distance,
            "opacity_gate": self.opacity_gate,
            "min_pull_opacity": self.min_pull_opacity,
            "warmup_iters": self.warmup_iters,
            "start_iter": self.start_iter,
            "stop_iter": self.stop_iter,
            "anchor_new_splats": self.anchor_new_splats,
            "teleport_threshold": self.teleport_threshold,
            "mode": self.mode,
            "reference_ply": self.reference_ply,
            "nn_refresh": self.nn_refresh,
            "stats_every": self.stats_every,
            "log_every": self.log_every,
            "stats_out": self.stats_out,
            "stats_snapshot_every": self.stats_snapshot_every,
        }

    _READONLY_CFG = ("free_radius_effective", "nn_spacing")

    def load_config(self, cfg):
        for k, v in (cfg or {}).items():
            if k in self._READONLY_CFG:
                continue  # reported for the record, not settable
            if hasattr(self, k) and not k.startswith("_") and not k.startswith("stat_"):
                setattr(self, k, v)
