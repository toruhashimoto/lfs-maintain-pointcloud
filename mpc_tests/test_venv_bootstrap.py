# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for locating the plugin's own venv from the standalone entry point.

LichtFeld Studio activates a plugin's `.venv` only when it loads that plugin.
A `--python-script` entry point instead gets the shared `~/.lichtfeld/venv`,
which does not carry this plugin's numpy/scipy. Without them the anchor does
not fail -- it silently falls back to `nn_spacing = 0`, i.e. no dead zone,
which is the exact configuration the dead zone exists to prevent.

Run:  python -m pytest mpc_tests/  (or plain: python mpc_tests/test_venv_bootstrap.py)
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venv_bootstrap import find_venv_site_packages  # noqa: E402


def _tree(root, *parts):
    path = os.path.join(root, *parts)
    os.makedirs(path)
    return path


def test_returns_none_without_a_venv():
    root = tempfile.mkdtemp()
    try:
        assert find_venv_site_packages(root) is None
    finally:
        shutil.rmtree(root)


def test_finds_windows_layout():
    root = tempfile.mkdtemp()
    try:
        want = _tree(root, ".venv", "Lib", "site-packages")
        assert find_venv_site_packages(root) == want
    finally:
        shutil.rmtree(root)


def test_finds_posix_layout():
    root = tempfile.mkdtemp()
    try:
        want = _tree(root, ".venv", "lib", "python3.12", "site-packages")
        assert find_venv_site_packages(root) == want
    finally:
        shutil.rmtree(root)


def test_ignores_a_venv_without_site_packages():
    # A half-created venv must not be reported as usable.
    root = tempfile.mkdtemp()
    try:
        _tree(root, ".venv", "Scripts")
        assert find_venv_site_packages(root) is None
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
