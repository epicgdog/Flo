import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker.metrics import bone_err, mpjpe_mm, pa_mpjpe_mm  # noqa: E402


def _poses(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.2, (n, 17, 3))


def _rot_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def test_identical_poses_mpjpe_zero():
    p = _poses()
    assert abs(mpjpe_mm(p, p)) < 1e-9


def test_10mm_shift_gives_10():
    p = _poses()
    shifted = p + np.array([0.01, 0.0, 0.0])
    assert abs(mpjpe_mm(shifted, p) - 10.0) < 1e-6


def test_rotated_copy_pa_mpjpe_under_0_1():
    g = _poses()
    rotated = g @ _rot_z(37.0).T
    assert pa_mpjpe_mm(rotated, g) < 0.1


def test_bone_err_zero_for_identical():
    p = _poses()
    assert bone_err(p, p) == 0.0
