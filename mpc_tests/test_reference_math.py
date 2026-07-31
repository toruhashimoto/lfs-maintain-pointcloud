# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the numpy reference of the anchor pull math.

Run:  python -m pytest mpc_tests/  (or plain: python mpc_tests/test_reference_math.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_math import anchor_pull, detect_relocations  # noqa: E402


def _ones_mask(n):
    return np.ones((n, 1), dtype=np.float64)


def test_zero_strength_is_identity():
    means = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(2), strength=0.0)
    np.testing.assert_allclose(out, means)


def test_at_anchor_no_pull():
    means = np.array([[1.0, 1.0, 1.0]])
    out = anchor_pull(means, means.copy(), _ones_mask(1), strength=0.5)
    np.testing.assert_allclose(out, means)


def test_pull_direction_and_magnitude():
    # 10 units away along +x, strength 0.1 -> moves 1 unit toward anchor.
    means = np.array([[10.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(1), strength=0.1)
    np.testing.assert_allclose(out, [[9.0, 0.0, 0.0]], atol=1e-9)


def test_hard_leash_snaps_to_anchor():
    # strength 1, no dead zone, no huber -> full snap back to the anchor.
    means = np.array([[3.0, 4.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(1), strength=1.0)
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0]], atol=1e-9)


def test_free_radius_dead_zone():
    # Inside the dead zone: no pull at all.
    means = np.array([[0.5, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(1), strength=1.0, free_radius=1.0)
    np.testing.assert_allclose(out, means)


def test_free_radius_snaps_to_boundary_not_center():
    # strength 1 + dead zone: projection onto the free-radius ball surface.
    means = np.array([[2.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(1), strength=1.0, free_radius=0.5)
    np.testing.assert_allclose(out, [[0.5, 0.0, 0.0]], atol=1e-9)


def test_huber_delta_bounds_pull():
    # excess 10, huber 0.2, strength 1 -> pull exactly 0.2.
    means = np.array([[10.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(1), strength=1.0, huber_delta=0.2)
    np.testing.assert_allclose(out, [[9.8, 0.0, 0.0]], atol=1e-9)


def test_max_distance_detaches_far_rows():
    means = np.array([[100.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    out = anchor_pull(means, anchors, _ones_mask(2), strength=0.5, max_distance=10.0)
    np.testing.assert_allclose(out[0], [100.0, 0.0, 0.0])  # untouched
    np.testing.assert_allclose(out[1], [0.5, 0.0, 0.0])    # pulled


def test_mask_zero_rows_untouched():
    means = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    mask = np.array([[1.0], [0.0]])
    out = anchor_pull(means, anchors, mask, strength=0.2)
    np.testing.assert_allclose(out[0], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [5.0, 0.0, 0.0])


def test_opacity_gate_scales_pull():
    # Very negative logit -> sigmoid ~ 0 -> almost no pull.
    means = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    op = np.array([[-20.0], [20.0]])
    out = anchor_pull(means, anchors, _ones_mask(2), strength=0.1,
                      opacity=op, opacity_gate=True)
    np.testing.assert_allclose(out[0], [10.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [9.0, 0.0, 0.0], atol=1e-6)


def test_soft_spring_converges_geometrically():
    # Repeated application decays the distance by (1-s) each step.
    means = np.array([[8.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    for _ in range(50):
        means = anchor_pull(means, anchors, _ones_mask(1), strength=0.2)
    assert np.linalg.norm(means) < 8.0 * (0.8 ** 50) + 1e-9


def test_relocation_detected_for_alive_jump():
    prev = np.zeros((3, 3))
    means = np.array([[0.0, 0.0, 0.001],   # noise-level jitter
                      [5.0, 0.0, 0.0],     # relocation (alive)
                      [0.0, 0.0, 0.0]])
    op = np.full((3, 1), 2.0)              # all alive
    reloc = detect_relocations(means, prev, op, op, threshold=0.1)
    assert reloc.flatten().tolist() == [False, True, False]


def test_noise_jump_on_dead_row_not_a_relocation():
    # A near-dead row kicked past the threshold by MCMC noise must NOT be
    # re-anchored (opacity stays near zero and unchanged).
    prev = np.zeros((1, 3))
    means = np.array([[5.0, 0.0, 0.0]])
    op_dead = np.full((1, 1), -6.0)        # sigmoid ~ 0.0025 < 0.02
    reloc = detect_relocations(means, prev, op_dead, op_dead, threshold=0.1)
    assert reloc.flatten().tolist() == [False]


def test_subthreshold_relocation_caught_by_opacity_revive():
    # Relocation that lands within the positional threshold is still
    # detected by the dead->alive opacity discontinuity.
    prev = np.zeros((1, 3))
    means = np.array([[0.05, 0.0, 0.0]])   # below threshold 0.1
    op_prev = np.full((1, 1), -6.0)        # sigmoid ~ 0.0025 (dead)
    op_cur = np.full((1, 1), 0.0)          # sigmoid 0.5 (alive)
    reloc = detect_relocations(means, prev, op_cur, op_prev, threshold=0.1)
    assert reloc.flatten().tolist() == [True]


def test_min_pull_opacity_disables_pull_on_invisible_rows():
    means = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    anchors = np.zeros_like(means)
    op = np.array([[-6.0], [2.0]])         # dead, alive
    out = anchor_pull(means, anchors, _ones_mask(2), strength=0.1,
                      opacity=op, min_pull_opacity=0.02)
    np.testing.assert_allclose(out[0], [10.0, 0.0, 0.0])   # untouched
    np.testing.assert_allclose(out[1], [9.0, 0.0, 0.0])    # pulled


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


def test_control_rows_are_never_pulled():
    # Control rows estimate the FREE drift distribution. If any setting
    # could pull them, the calibration input would be measuring the leash's
    # own output and the feedback loop would collapse the dead zone.
    means = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    anchors = np.zeros((2, 3))
    control = np.array([[0.0], [1.0]])
    out = anchor_pull(means, anchors, _ones_mask(2), strength=1.0,
                      control=control)
    assert not np.allclose(out[0], means[0])      # normal row moved
    np.testing.assert_allclose(out[1], means[1])  # control row untouched


def test_control_none_matches_zero_control():
    means = np.array([[3.0, 4.0, 0.0]])
    anchors = np.zeros((1, 3))
    a = anchor_pull(means, anchors, _ones_mask(1), strength=0.5)
    b = anchor_pull(means, anchors, _ones_mask(1), strength=0.5,
                    control=np.zeros((1, 1)))
    np.testing.assert_allclose(a, b)
