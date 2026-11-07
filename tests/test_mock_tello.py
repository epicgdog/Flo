import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.mock_tello import MockTello  # noqa: E402
from harness import config as C  # noqa: E402


@pytest.fixture
def tello():
    t = MockTello(cfg=C.load())
    t.connect()
    return t


def test_takeoff_then_move_up(tello):
    tello.takeoff()
    tello.move_up(100)
    assert tello.get_height() == 180


def test_rotate_clockwise_changes_yaw_by_90(tello):
    before = tello.get_yaw()
    tello.rotate_clockwise(90)
    assert (tello.get_yaw() - before) % 360 == 90


def test_height_never_exceeds_limit(tello):
    lo, hi = C.load()["sim"]["height_limit_cm"]
    tello.takeoff()
    tello.move_up(10000)
    assert tello.get_height() == hi
    tello.move_down(10000)
    assert tello.get_height() == lo


def test_unsupported_method_raises(tello):
    with pytest.raises(NotImplementedError):
        tello.get_acceleration_x()
