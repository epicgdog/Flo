"""Compare our lifter with FMPose3D (CVPR 2026) on the dev and held-out eval sets.

Three steps, because FMPose3D needs Python 3.10 and torch<2.5 (kept out of the project venv):
  uv run python3 scripts/compare_fmpose3d.py ours [--tag latest]
  uv run --no-project --python 3.10 --with fmpose3d --with numpy python3 scripts/compare_fmpose3d.py fmpose
  uv run python3 scripts/compare_fmpose3d.py report

Caveats (also printed by `report`):
- FMPose3D predicts H36M-17 joints; scoring uses the 13 joints both skeletons share (pelvis root + 12 limbs).
- FMPose3D was trained on Human3.6M S1,5,6,7,8. H3WB is Human3.6M, and our held-out set is S8, so FMPose3D has
  seen these poses (from H36M's cameras, not drone views).
- FMPose3D has no visibility mask: dropped joints are filled with their noiseless projection (favours FMPose3D).
  The `visible` rows use only samples with no dropped joints.
- Pixel keypoints are rebuilt from the normalized eval kp2d and frame. Joints outside the image are filled
  with off-image projections, which FMPose3D never saw in training.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs", "compare")
SETS = {"dev": "data/dev_eval.npz", "heldout": "data/heldout_eval.npz"}
W, H = 960, 720      # data.sensor_px

# COCO-17 -> H36M-17. Composite H36M joints are built from COCO joints.
#   H36M: 0 pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle, 7 spine, 8 thorax, 9 neck/nose,
#         10 head, 11 LShoulder, 12 LElbow, 13 LWrist, 14 RShoulder, 15 RElbow, 16 RWrist
H36M_FROM_COCO = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}
SHARED_H36M = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
SHARED_COCO = [None, 12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10]       # None = pelvis (mean of hips 11, 12)


def coco_to_h36m(p: np.ndarray) -> np.ndarray:
    """(..., 17, D) COCO -> (..., 17, D) H36M."""
    out = np.zeros_like(p)
    pelvis = (p[..., 11, :] + p[..., 12, :]) / 2
    thorax = (p[..., 5, :] + p[..., 6, :]) / 2
    out[..., 0, :] = pelvis
    out[..., 7, :] = (pelvis + thorax) / 2
    out[..., 8, :] = thorax
    out[..., 10, :] = p[..., 1:5, :].mean(axis=-2)
    for h, c in H36M_FROM_COCO.items():
        out[..., h, :] = p[..., c, :]
    return out


def shared_from_coco(p: np.ndarray) -> np.ndarray:
    pelvis = (p[..., 11, :] + p[..., 12, :]) / 2
    return np.stack([pelvis] + [p[..., c, :] for c in SHARED_COCO[1:]], axis=-2)


def pixels(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild pixel keypoints: (N,17,2) with eval noise on visible joints, noiseless fill for dropped ones.
    frame = (root_u - W/2)/f, (root_v - H/2)/f, diag/f, so visible joints come back exactly."""
    from checker.datagen import camera_extrinsics
    kp, mask, frame, cam = d["kp2d"].astype(np.float64), d["mask"], d["frame"].astype(np.float64), d["cam"]
    f = cam[:, 3].astype(np.float64)
    root = np.stack([frame[:, 0] * f + W / 2, frame[:, 1] * f + H / 2], axis=1)
    noisy = kp * (frame[:, 2] * f)[:, None, None] + root[:, None, :]
    R, pos = camera_extrinsics(cam)
    root_cam = np.einsum("nij,nj->ni", R, -pos)                     # world root (origin) in camera coordinates
    X = d["pose3d"].astype(np.float64) + root_cam[:, None, :]
    z = np.maximum(X[..., 2], 1e-6)
    clean = np.stack([f[:, None] * X[..., 0] / z + W / 2, f[:, None] * X[..., 1] / z + H / 2], axis=-1)
    vis = mask > 0
    return np.where(vis[..., None], noisy, clean).astype(np.float32), vis


def cmd_ours(tag: str) -> None:
    from checker.check import load_model, predict
    from harness import config as C
    from sim.lifter import model_dir_for_tag
    cfg = C.load()
    model, _ = load_model(model_dir_for_tag(tag, cfg))
    os.makedirs(OUT, exist_ok=True)
    for name, path in SETS.items():
        d = np.load(os.path.join(ROOT, path))
        hyps = predict(model, d["kp2d"], d["mask"], d["frame"], cfg)
        np.save(os.path.join(OUT, f"ours_{name}.npy"), hyps.mean(axis=1).astype(np.float32))
        print(f"ours ({tag}) {name}: {hyps.shape[0]} predictions")
    json.dump({"tag": tag}, open(os.path.join(OUT, "ours_meta.json"), "w"))


def cmd_fmpose() -> None:
    from fmpose3d import FMPose3DInference
    api = FMPose3DInference()
    os.makedirs(OUT, exist_ok=True)
    for name, path in SETS.items():
        d = dict(np.load(os.path.join(ROOT, path)))
        px, _ = pixels(d)
        res = api.pose_3d(coco_to_h36m(px), (H, W), camera_rotation=None, seed=0)
        np.save(os.path.join(OUT, f"fmpose_{name}.npy"), np.asarray(res.poses_3d, dtype=np.float32))
        print(f"fmpose3d {name}: {np.asarray(res.poses_3d).shape}")


def _mpjpe(p, g):
    return float(np.linalg.norm(p - g, axis=-1).mean() * 1000)


def _pa_mpjpe(p, g):
    X, Y = p - p.mean(1, keepdims=True), g - g.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(np.einsum("nki,nkj->nij", Y, X))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.stack([np.ones_like(d), np.ones_like(d), d], 1)
    R = (U * D[:, None, :]) @ Vt
    s = (S * D).sum(1) / (X ** 2).sum((1, 2))
    al = s[:, None, None] * np.einsum("nki,nji->nkj", X, R) + g.mean(1, keepdims=True)
    return float(np.linalg.norm(al - g, axis=-1).mean() * 1000)


def cmd_report() -> None:
    tag = json.load(open(os.path.join(OUT, "ours_meta.json")))["tag"]
    rows = []
    for name, path in SETS.items():
        d = np.load(os.path.join(ROOT, path))
        gt = shared_from_coco(d["pose3d"].astype(np.float64))
        ours = shared_from_coco(np.load(os.path.join(OUT, f"ours_{name}.npy")).astype(np.float64))
        fm = np.load(os.path.join(OUT, f"fmpose_{name}.npy")).astype(np.float64)
        fm = fm.reshape(len(gt), 17, 3)[:, SHARED_H36M]
        fm = fm - fm[:, :1]                                     # pelvis-relative, like ours
        allvis = (d["mask"] > 0).all(1)
        for subset, sel in (("all", np.ones(len(gt), bool)), ("visible", allvis)):
            for model, pred in ((f"ours {tag}", ours), ("FMPose3D", fm)):
                rows.append({"set": name, "subset": subset, "n": int(sel.sum()), "model": model,
                             "mpjpe_mm": _mpjpe(pred[sel], gt[sel]), "pa_mpjpe_mm": _pa_mpjpe(pred[sel], gt[sel])})
    json.dump(rows, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    print(f"{'set':8} {'subset':8} {'n':>5}  {'model':12} {'MPJPE':>8} {'PA-MPJPE':>9}   (13 shared joints, mm)")
    for r in rows:
        print(f"{r['set']:8} {r['subset']:8} {r['n']:>5}  {r['model']:12} {r['mpjpe_mm']:8.1f} {r['pa_mpjpe_mm']:9.1f}")
    print("\n" + __doc__.split("Caveats")[1].strip())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "ours":
        cmd_ours(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--tag" else "latest")
    elif mode == "fmpose":
        cmd_fmpose()
    elif mode == "report":
        cmd_report()
    else:
        raise SystemExit(__doc__)
