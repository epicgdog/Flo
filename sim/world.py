"""Virtual world: a person at the origin, a ground grid, and a forward-facing camera pitched down.

Pose = a random held-out test pose (world frame, root-relative, z-up), rotated by a random yaw.
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.datagen import BONES  # noqa: E402


class World:
    def __init__(self, poses: np.ndarray, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.poses = poses.astype(np.float64)               # (M,17,3) meters, z-up, root-relative
        self.rng = np.random.default_rng(seed)
        s = cfg["sim"]
        self.W, self.H = s["sensor_px"]
        self.focal = float(s["focal_px"])
        self.pitch = math.radians(s["camera_pitch_down_deg"])
        self.change_every = int(s["pose_change_every_frames"])
        self.frame_idx = 0
        self._new_pose()

    def _new_pose(self):
        self.pose = self.poses[self.rng.integers(0, len(self.poses))]
        lo, hi = self.cfg["sim"]["person_yaw_deg"]
        a = math.radians(self.rng.uniform(lo, hi))
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        self.pose_world = self.pose @ R.T

    def _camera(self, drone: dict):
        """Camera position (m), forward/right/down axes. Heading is the same convention MockTello moves by."""
        yaw = math.radians(drone["yaw"])
        pos = np.array([drone["x"], drone["y"], drone["z"]])
        p = self.pitch
        fwd = np.array([math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p)])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return pos, fwd, right, down

    def render(self, drone: dict, pose=None):
        """-> (frame RGB uint8 (H,W,3), kp2d_px (17,2), mask (17,), gt_pose3d (17,3) camera frame, root-relative)."""
        if pose is not None:
            self.pose_world = pose
        pos, fwd, right, down = self._camera(drone)
        J = self.pose_world                                    # world joints, person at origin
        rel = J - pos
        cam = np.stack([rel @ right, rel @ down, rel @ fwd], axis=1)   # (17,3) camera frame
        z = np.maximum(cam[:, 2], 1e-6)
        u = self.focal * cam[:, 0] / z + self.W / 2
        v = self.focal * cam[:, 1] / z + self.H / 2
        uv = np.stack([u, v], axis=1)
        mask = ((cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H))
        gt = cam - cam[[11, 12]].mean(axis=0)

        frame = np.full((self.H, self.W, 3), 40, dtype=np.uint8)
        self._draw_grid(frame, pos, right, down, fwd)
        for a, b in BONES:
            if mask[a] and mask[b]:
                cv2.line(frame, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), (255, 255, 255), 3)
        for i in range(17):
            if mask[i]:
                cv2.circle(frame, (int(uv[i, 0]), int(uv[i, 1])), 4, (255, 80, 80), -1)

        self.frame_idx += 1
        if self.frame_idx % self.change_every == 0:
            self._new_pose()
        return frame, uv.astype(np.float32), mask.astype(np.float32), gt.astype(np.float32)

    def _draw_grid(self, frame, pos, right, down, fwd):
        for g in np.arange(-6, 7, 1.0):
            for a, b in ((np.array([g, -6, 0.0]), np.array([g, 6, 0.0])), (np.array([-6, g, 0.0]), np.array([6, g, 0.0]))):
                pts = []
                for q in (a, b):
                    r = q - pos
                    zc = r @ fwd
                    if zc <= 0.05:
                        pts = None
                        break
                    pts.append((int(self.focal * (r @ right) / zc + self.W / 2),
                                int(self.focal * (r @ down) / zc + self.H / 2)))
                if pts:
                    cv2.line(frame, pts[0], pts[1], (90, 90, 90), 1)
