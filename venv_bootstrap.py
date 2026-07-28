# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locate the plugin's own virtualenv from the standalone entry point.

Deliberately free of any ``lichtfeld`` import so it can be unit-tested
without the embedded interpreter.
"""


import glob
import os


def find_venv_site_packages(root):
    """Return the site-packages dir of ``root/.venv``, or None.

    Checks the Windows layout first, then any POSIX ``lib/pythonX.Y`` one.
    A ``.venv`` that exists but has no site-packages counts as absent, so a
    half-created environment is never put on ``sys.path``.
    """
    venv = os.path.join(root, ".venv")
    candidates = [os.path.join(venv, "Lib", "site-packages")]
    candidates += sorted(
        glob.glob(os.path.join(venv, "lib", "python*", "site-packages")))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None
