# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""maintain_pointcloud - soft position anchor plugin for LichtFeld Studio.

Keeps 3DGS splats close to the pre-placed initial point cloud during
training without hard-freezing them ("keep the init cloud" regularizer).
"""

import lichtfeld as lf

from .anchor_core import AnchorRegularizer, apply_env_overrides
from .panels.anchor_panel import AnchorPanel

regularizer = apply_env_overrides(AnchorRegularizer())

_handler = None
_classes = [AnchorPanel]


def on_load():
    global _handler
    AnchorPanel.regularizer = regularizer
    for cls in _classes:
        lf.register_class(cls)
    _handler = lf.ScopedHandler()
    _handler.on_training_start(regularizer.on_training_start)
    _handler.on_post_step(regularizer.apply)
    _handler.on_training_end(regularizer.on_training_end)
    lf.log.info("maintain_pointcloud plugin loaded")


def on_unload():
    global _handler
    if _handler is not None:
        _handler.clear()
        _handler = None
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    # Drop GPU buffers immediately instead of waiting for a cyclic GC pass
    # (the module dict / panel class / regularizer form a reference cycle).
    AnchorPanel.regularizer = None
    regularizer.reset()
    lf.log.info("maintain_pointcloud plugin unloaded")
