# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pins the shipped defaults that carry a decision.

``lichtfeld`` exists only inside the host application, so importing
anchor_core here needs an import-time stub. Nothing about the stub is
exercised: ``AnchorRegularizer.__init__`` is pure field initialisation,
and the assertions read the real defaults off the real class.

Run:  .venv\\Scripts\\python.exe -m pytest mpc_tests/
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("lichtfeld", types.ModuleType("lichtfeld"))
import anchor_core  # noqa: E402


def test_default_calibrate_quantile_is_60():
    # Decided 2026-07-31 (owner approval) on two agreeing datasets: q60
    # tightens drift p95 by a further 7-12% over q70 for +0.8pt scale_p90
    # and no photometric cost. q50 is NOT the default because it doubles
    # the shape-cost increment, shows the first statistically visible SSIM
    # deficit vs q70 (C, effect/noise 5.0), and sits on the edge of the
    # swept range -- a default belongs where both neighbours are measured.
    assert anchor_core.AnchorRegularizer().calibrate_quantile == 60.0


def test_default_enabled_is_false():
    # The positioning commitment: the anchor stays opt-in. Enabling it is
    # a per-delivery decision (position fidelity for measurement/overlay),
    # never something an update turns on behind the operator's back.
    assert anchor_core.AnchorRegularizer().enabled is False
