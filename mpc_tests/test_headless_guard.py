# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""headless_anchor must not double-register when the plugin is loaded.

With ``load_on_startup: true`` (GUI use) a headless campaign run that also
passes ``--python-script headless_anchor.py`` would otherwise register a
SECOND AnchorRegularizer: plugins load before the script runs, both hook
post_step, and every splat gets pulled twice per iteration. The guard makes
the script detect the loaded plugin and reuse its regularizer instead.

``lichtfeld`` is stubbed at import time only (log + hook decorators that
record what gets registered); the decision logic under test is real.
"""

import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Log:
    def info(self, *_a, **_k):
        pass

    error = warning = info


def _lf_stub(registered):
    lf = types.ModuleType("lichtfeld")
    lf.log = _Log()

    def make(name):
        def deco(fn):
            registered.append(name)
            return fn
        return deco

    lf.on_training_start = make("start")
    lf.on_post_step = make("post_step")
    lf.on_training_end = make("end")
    return lf


def _import_headless(monkeypatch, plugin_module):
    registered = []
    monkeypatch.setitem(sys.modules, "lichtfeld", _lf_stub(registered))
    if plugin_module is None:
        monkeypatch.delitem(sys.modules, "maintain_pointcloud", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "maintain_pointcloud", plugin_module)
    sys.modules.pop("headless_anchor", None)
    mod = importlib.import_module("headless_anchor")
    return mod, registered


def test_standalone_registers_its_own_hooks(monkeypatch):
    mod, registered = _import_headless(monkeypatch, plugin_module=None)
    assert registered == ["start", "post_step", "end"]
    assert mod.reg is not None


def test_reuses_the_loaded_plugins_regularizer_without_new_hooks(monkeypatch):
    import anchor_core

    plugin = types.ModuleType("maintain_pointcloud")
    plugin.regularizer = anchor_core.AnchorRegularizer()
    mod, registered = _import_headless(monkeypatch, plugin_module=plugin)
    # The plugin's ScopedHandler already owns the hooks; a second set here
    # would pull every splat twice per iteration.
    assert registered == []
    assert mod.reg is plugin.regularizer
    # Campaigns configure through the environment; the reused instance must
    # have gone through the same env-override pass as the standalone path.
    assert mod.reg.enabled is True
