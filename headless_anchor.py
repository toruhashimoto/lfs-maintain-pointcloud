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

from anchor_core import AnchorRegularizer, apply_env_overrides  # noqa: E402

reg = AnchorRegularizer()
reg.enabled = True  # headless default: on (override with LFS_MPC_ENABLED=0)
apply_env_overrides(reg)
lf.log.info("[maintain_pointcloud/headless] config: " + json.dumps(reg.config_dict()))


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
