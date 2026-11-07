"""Loads a tagged model and lifts 2D keypoints (pixels) to 3D."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.check import load_model  # noqa: E402
from checker.sampler import sample_hypotheses  # noqa: E402
from harness import config as C  # noqa: E402
from harness import history  # noqa: E402


def model_dir_for_tag(tag: str, cfg: dict) -> str:
    """history/<iter of tag>/model/. 'latest' = the newest accepted tag."""
    tagged = [(o["tag"], o["iter_id"]) for o in history.outcomes(cfg) if o.get("accepted") and o.get("tag")]
    if not tagged:
        raise SystemExit("no tagged models yet: run checker/calibrate.py (M4)")
    if tag == "latest":
        tag = max(tagged, key=lambda t: int(t[0][1:]))[0]
    for t, iter_id in tagged:
        if t == tag:
            return os.path.join(C.abspath(cfg["paths"]["history"]), iter_id, "model")
    raise SystemExit(f"no model for tag {tag}")


def normalize_kp(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exactly datagen.project step 7: subtract the 2D root (mean of hips; mean of visible joints if both hips
    are dropped), divide by the bounding-box diagonal of visible joints, zero the dropped joints."""
    vis = mask > 0
    if vis[11] or vis[12]:
        hips = [j for j in (11, 12) if vis[j]]
        root = uv[hips].mean(axis=0)
    else:
        root = uv[vis].mean(axis=0) if vis.any() else np.zeros(2)
    diag = float(np.linalg.norm(uv[vis].max(axis=0) - uv[vis].min(axis=0))) if vis.any() else 1.0
    if not np.isfinite(diag) or diag <= 1e-6:
        diag = 1.0
    return ((uv - root) / diag * mask[:, None]).astype(np.float32)


class Lifter:
    def __init__(self, tag: str, cfg: dict):
        """Loads history/<iter of tag>/model/."""
        self.cfg = cfg
        self.tag = tag
        self.model, _ = load_model(model_dir_for_tag(tag, cfg))

    def lift(self, kp2d_px, mask) -> tuple[np.ndarray, np.ndarray]:
        """Normalizes exactly like datagen.project step 7; returns (pred (17,3), hyps (K,17,3))."""
        mask = np.asarray(mask, dtype=np.float32)
        kp = normalize_kp(np.asarray(kp2d_px, dtype=np.float64), mask)
        cond = torch.from_numpy(np.concatenate([kp.reshape(-1), mask])[None].astype(np.float32))
        s = self.cfg["sample"]
        with torch.no_grad():
            h = sample_hypotheses(self.model, cond, s["k_hypotheses"], s["euler_steps"], s["noise_seed"])
        hyps = h[0].reshape(-1, 17, 3).double().numpy()
        return hyps.mean(axis=0), hyps
