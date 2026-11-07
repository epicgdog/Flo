"""Mock DJI Tello with the djitellopy method names and signatures. Unsupported methods raise NotImplementedError.

Body frame: x forward, y left, z up. Yaw in degrees, world frame, clockwise = positive.
Internal units: cm for position, degrees for yaw. sim_state() returns meters.
"""
import math
import types

import numpy as np

from harness import config as C

SUPPORTED = {
    "connect", "end", "get_battery", "takeoff", "land", "streamon", "streamoff", "get_frame_read",
    "move_up", "move_down", "move_left", "move_right", "move_forward", "move_back",
    "rotate_clockwise", "rotate_counter_clockwise", "go_xyz_speed", "send_rc_control",
    "get_height", "get_yaw", "sim_state",
}


class MockTello:
    def __init__(self, world=None, cfg: dict | None = None):
        self.cfg = cfg or C.load()
        self.world = world                      # object with render(drone_state, pose) -> (frame, kp2d, mask, gt)
        self.height_min, self.height_max = self.cfg["sim"]["height_limit_cm"]
        self.fps = self.cfg["sim"]["fps"]
        self.connected = False
        self.streaming = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.flight_s = 0.0
        self.rc = (0, 0, 0, 0)
        self.last_render = None
        self.frame_read = types.SimpleNamespace(frame=np.zeros((720, 960, 3), dtype=np.uint8))

    def __getattr__(self, name):
        # Only reached for attributes not defined on the class.
        if name in SUPPORTED or name.startswith("_"):
            raise AttributeError(name)
        raise NotImplementedError(f"MockTello does not implement {name}")

    # connection and video
    def connect(self):
        self.connected = True

    def end(self):
        self.connected = False

    def streamon(self):
        self.streaming = True

    def streamoff(self):
        self.streaming = False

    def get_frame_read(self):
        self._render()
        return self.frame_read

    # flight
    def get_battery(self) -> int:
        return max(0, 100 - int(self.flight_s // 30))

    def takeoff(self):
        self.z = 80.0
        self.flight_s += 3.0

    def land(self):
        self.z = 0.0
        self.flight_s += 3.0

    def _move_body(self, fwd=0.0, left=0.0, up=0.0):
        yaw = math.radians(self.yaw)
        # body x forward, y left; world yaw measured clockwise
        dx = fwd * math.cos(yaw) - left * math.sin(yaw)
        dy = fwd * math.sin(yaw) + left * math.cos(yaw)
        self.x += dx
        self.y += dy
        if self.z > 0:  # airborne: keep inside sim.height_limit_cm [min, max]
            self.z = min(max(self.z + up, float(self.height_min)), float(self.height_max))
        self.flight_s += (abs(fwd) + abs(left) + abs(up)) / 50.0 + 0.1

    def move_forward(self, x: int):
        self._move_body(fwd=x)

    def move_back(self, x: int):
        self._move_body(fwd=-x)

    def move_left(self, x: int):
        self._move_body(left=x)

    def move_right(self, x: int):
        self._move_body(left=-x)

    def move_up(self, x: int):
        self._move_body(up=x)

    def move_down(self, x: int):
        self._move_body(up=-x)

    def rotate_clockwise(self, x: int):
        self.yaw = (self.yaw + x) % 360.0
        self.flight_s += 0.5

    def rotate_counter_clockwise(self, x: int):
        self.yaw = (self.yaw - x) % 360.0
        self.flight_s += 0.5

    def go_xyz_speed(self, x: int, y: int, z: int, speed: int):
        self._move_body(fwd=x, left=y, up=z)

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        """Velocity command, applied on each tick() at fps."""
        self.rc = (lr, fb, ud, yaw)

    def tick(self, dt: float | None = None):
        """Advance the rc velocity command by one simulation step."""
        dt = dt if dt is not None else 1.0 / self.fps
        lr, fb, ud, yaw = self.rc
        if lr or fb or ud or yaw:
            self._move_body(fwd=fb * dt, left=lr * dt, up=ud * dt)
            self.yaw = (self.yaw + yaw * dt) % 360.0

    # state
    def get_height(self) -> int:
        return int(round(self.z))

    def get_yaw(self) -> int:
        return int(round(self.yaw))

    def sim_state(self) -> dict:
        return {"x": self.x / 100.0, "y": self.y / 100.0, "z": self.z / 100.0, "yaw": self.yaw}

    def _render(self):
        if self.world is None or not self.streaming:
            return
        self.last_render = self.world.render(self.sim_state(), None)
        self.frame_read.frame = self.last_render[0]
