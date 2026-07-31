# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless entry point for the maintain_pointcloud anchor regularizer.

Standalone --python-script variant (no plugin install needed):

    LichtFeld-Studio.exe --train -d <dataset> -o <out> \
        --python-script path/to/headless_anchor.py

Configuration via LFS_MPC_* environment variables -- see
anchor_core.apply_env_overrides for the full list. Notable:

    LFS_MPC_ENABLED     "1"/"0" (default 1 here, unlike the GUI plugin)
    LFS_MPC_STRENGTH    float, e.g. "0.1"
    LFS_MPC_STATS_OUT   path: write final init-drift stats JSON on training end
    LFS_MPC_CONFIG      path to a JSON file with AnchorRegularizer params
"""

import json
import os
import sys

import lichtfeld as lf

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from venv_bootstrap import find_venv_site_packages  # noqa: E402

# LichtFeld Studio activates a plugin's own .venv only when it *loads that
# plugin*. Run as --python-script the interpreter gets the shared
# ~/.lichtfeld/venv instead, which does not carry this plugin's numpy/scipy.
_site = find_venv_site_packages(_here)
if _site and _site not in sys.path:
    sys.path.insert(0, _site)

from anchor_core import AnchorRegularizer, apply_env_overrides  # noqa: E402

# Missing numpy/scipy does not raise: capture catches it and falls back to
# scene_diag = 0 (teleport detection off) and nn_spacing = 0 (no dead zone).
# A fixed strength with no dead zone is exactly the hard-freeze failure mode
# the dead zone exists to prevent, so say so loudly rather than degrade.
_missing = []
try:
    import numpy  # noqa: F401
except ImportError:
    _missing.append("numpy (scene diagonal + teleport detection + drift stats)")
try:
    from scipy.spatial import cKDTree  # noqa: F401
except ImportError:
    _missing.append("scipy (automatic dead zone from point spacing)")
if _missing:
    lf.log.error(
        "[maintain_pointcloud/headless] MISSING DEPENDENCIES: "
        + "; ".join(_missing)
        + f". Searched {_site or _here + os.sep + '.venv (absent)'}. "
        "The anchor will run with free_radius=0, which turns a fixed "
        "strength into a hard freeze late in training. Install them into "
        "the plugin venv or ~/.lichtfeld/venv before trusting this run."
    )

# Plugins load BEFORE --python-script runs. If the installed plugin is
# already active (load_on_startup: true for GUI use), registering a second
# regularizer here would pull every splat twice per iteration -- so reuse
# the plugin's instance and let its ScopedHandler keep owning the hooks.
_plugin = sys.modules.get("maintain_pointcloud")
_plugin_reg = getattr(_plugin, "regularizer", None) if _plugin else None

if _plugin_reg is not None:
    reg = _plugin_reg
    reg.enabled = True  # headless default: on (override with LFS_MPC_ENABLED=0)
    apply_env_overrides(reg)
    lf.log.info(
        "[maintain_pointcloud/headless] plugin already loaded; reusing its "
        "regularizer (no extra hooks). config: " + json.dumps(reg.config_dict()))
else:
    reg = AnchorRegularizer()
    reg.enabled = True  # headless default: on (override with LFS_MPC_ENABLED=0)
    apply_env_overrides(reg)
    lf.log.info("[maintain_pointcloud/headless] config: "
                + json.dumps(reg.config_dict()))

    @lf.on_training_start
    def _on_start(hook):
        reg.on_training_start(hook)

    @lf.on_post_step
    def _on_post_step(hook):
        reg.apply(hook)

    @lf.on_training_end
    def _on_end(hook):
        reg.on_training_end(hook)

    lf.log.info("[maintain_pointcloud/headless] hooks registered")
