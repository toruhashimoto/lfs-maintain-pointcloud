# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings panel for the maintain_pointcloud plugin."""

import lichtfeld as lf

_MODES = ["index", "nn"]


class AnchorPanel(lf.ui.Panel):
    """Position-anchor controls (custom training parameters)."""

    id = "maintain_pointcloud.anchor_panel"
    label = "PointCloud Anchor"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220

    # Injected by __init__.on_load (shared AnchorRegularizer instance).
    regularizer = None

    def draw(self, ui):
        reg = self.regularizer
        if reg is None:
            ui.text_wrapped("Regularizer not initialized.")
            return

        ui.heading("Maintain PointCloud")
        ui.text_wrapped(
            "Softly anchors splat positions to the initial point cloud "
            "during training. 0 strength / disabled = vanilla training."
        )

        _, reg.enabled = ui.checkbox("Enabled", reg.enabled)

        ui.separator()
        ui.text_disabled("Pull")
        _, reg.strength = ui.slider_float("Strength (0-1)", reg.strength, 0.0, 1.0)
        _, reg.free_radius = ui.input_float(
            "Free radius (world units, 0=auto)", reg.free_radius, 0.001, 0.01, "%.5f")
        _, reg.free_radius_spacing = ui.input_float(
            "  auto = N x cloud spacing", reg.free_radius_spacing, 0.1, 1.0, "%.2f")
        _, reg.huber_delta = ui.input_float(
            "Huber delta (max pull dist, 0=off)", reg.huber_delta, 0.001, 0.01, "%.5f")
        _, reg.max_distance = ui.input_float(
            "Max distance (0=inf)", reg.max_distance, 0.01, 0.1, "%.4f")
        _, reg.opacity_gate = ui.checkbox("Opacity gate", reg.opacity_gate)
        _, reg.min_pull_opacity = ui.input_float(
            "Min pull opacity (0=off)", reg.min_pull_opacity, 0.001, 0.01, "%.3f")

        ui.separator()
        ui.text_disabled("Schedule")
        _, reg.warmup_iters = ui.input_int("Warmup iters", reg.warmup_iters)
        _, reg.start_iter = ui.input_int("Start iter", reg.start_iter)
        _, reg.stop_iter = ui.input_int("Stop iter (0=never)", reg.stop_iter)

        ui.separator()
        ui.text_disabled("MCMC / topology")
        _, reg.anchor_new_splats = ui.checkbox(
            "Anchor new/relocated splats at birth", reg.anchor_new_splats)
        _, reg.teleport_threshold = ui.input_float(
            "Teleport threshold (0=auto 1% diag)", reg.teleport_threshold,
            0.001, 0.01, "%.5f")

        ui.separator()
        ui.text_disabled("Anchor mode")
        mode_idx = _MODES.index(reg.mode) if reg.mode in _MODES else 0
        changed, mode_idx = ui.combo("Mode", mode_idx, _MODES)
        if changed:
            reg.mode = _MODES[mode_idx]
        if reg.mode == "nn":
            _, reg.reference_ply = ui.input_text_with_hint(
                "Reference PLY", "(empty = init snapshot)", reg.reference_ply)
            _, reg.nn_refresh = ui.input_int("NN refresh iters", reg.nn_refresh)

        ui.separator()
        ui.text_disabled("Diagnostics")
        _, reg.log_every = ui.input_int("Log every N iters (0=off)", reg.log_every)
        # Buttons only set request flags; the training thread services them
        # at the next hook (all tensor-state mutation stays on one thread).
        if ui.button("Re-capture anchors now"):
            reg.request_capture()
            lf.log.info("maintain_pointcloud: re-capture requested")
        ui.same_line()
        if ui.button_styled("Reset state", "secondary"):
            reg.request_reset()
            lf.log.info("maintain_pointcloud: reset requested")

        ui.separator()
        if reg.stat_captured:
            ui.text_wrapped(
                f"free radius in use {reg._effective_free_radius():.5f} "
                f"(cloud spacing {reg._nn_spacing:.5f})\n"
                f"iter {reg.stat_iter} | anchored {reg.stat_anchored_rows}/{reg.stat_rows} rows\n"
                f"excess mean {reg.stat_mean_excess:.6f} / max {reg.stat_max_excess:.6f}\n"
                f"teleports {reg.stat_teleports} | appended {reg.stat_appended} | "
                f"applied {reg.stat_applied_iters} iters"
            )
        else:
            ui.text_disabled("Anchors not captured yet (start training).")
