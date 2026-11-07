"""2D keypoint source. The sim skips 2D detection: it adds noise to the projected joints."""
import numpy as np

_RNG = np.random.default_rng(0)


def sim_keypoints(world_out: tuple, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """projected joints + Gaussian noise sim.noise_px -> (kp2d_px (17,2), mask (17,)).
    world_out is World.render()'s (frame, kp2d_px, mask, gt_pose3d)."""
    _, joints_px, mask, _ = world_out
    kp = joints_px + _RNG.normal(0.0, cfg["sim"]["noise_px"], joints_px.shape)
    return kp.astype(np.float32), np.asarray(mask, dtype=np.float32)


def detector_keypoints(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real Tello only: any COCO-17 2D pose model; raises NotImplementedError until chosen."""
    raise NotImplementedError("no 2D detector chosen yet; real Tello flights need one")
