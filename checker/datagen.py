"""Synthetic drone views of 3D body poses. Poses are (N,17,3) meters, root-relative, z-up.

COCO-17 joint order. Root = mean of joints 11 (left hip) and 12 (right hip).
"""
import io
import json
import pickle
import zipfile

import numpy as np

BODY_JOINTS = list(range(17))          # COCO order
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 6), (11, 12), (5, 11), (6, 12), (0, 1), (0, 2), (1, 3), (2, 4)]   # 16 bones
EXTREMITIES = (9, 10, 15, 16)
ROOT_JOINTS = (11, 12)
_KEYS_3D = ("keypont_3d", "keypoints_3d")   # spec spelling, and the spelling in 2Dto3D_train.json


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The H3WB test npz is a pickled dict of arrays. Allow only the numpy classes it needs."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        if name == "_reconstruct":
            from numpy._core import multiarray
            return multiarray._reconstruct
        return getattr(np, name)


def _read_raw_json(path: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    out = []
    for sid in sorted(data, key=lambda k: int(k)):
        s = data[sid]
        kp = next(s[k] for k in _KEYS_3D if k in s)
        out.append([[kp[str(j)]["x"], kp[str(j)]["y"], kp[str(j)]["z"]] for j in BODY_JOINTS])
    return np.array(out, dtype=np.float64)


def _read_raw_npz(path: str) -> np.ndarray:
    """H3WB 2D->3D test set (task1_test_3d.npz): subject -> action -> camera -> camera_3d (n,133,3)."""
    raw = zipfile.ZipFile(path).read("data.npy")
    f = io.BytesIO(raw)
    ver = np.lib.format.read_magic(f)
    (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
    data = _NumpyOnlyUnpickler(f).load().item()
    out = []
    for subj in sorted(data):
        for action in sorted(data[subj]):
            entry = data[subj][action]
            for cam in sorted(k for k, v in entry.items() if isinstance(v, dict) and "camera_3d" in v):
                out.append(np.asarray(entry[cam]["camera_3d"])[:, :17, :])
    return np.concatenate(out).astype(np.float64)


def _rotation_to_z(v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector v to (0,0,1)."""
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, target)
    s = np.linalg.norm(axis)
    c = float(np.dot(v, target))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def load_h3wb_body(json_path: str, cfg: dict) -> np.ndarray:
    """(N,17,3): meters, root-relative (root = mean of joints 11,12), z-up."""
    poses = _read_raw_npz(json_path) if json_path.endswith(".npz") else _read_raw_json(json_path)
    poses = poses - poses[:, list(ROOT_JOINTS)].mean(axis=1, keepdims=True)
    if np.median(np.linalg.norm(poses[:, 5] - poses[:, 6], axis=1)) > 10:
        poses = poses / 1000.0
    up_vecs = (poses[:, 5] + poses[:, 6]) / 2.0 - poses[:, list(ROOT_JOINTS)].mean(axis=1)
    up_vecs = up_vecs / np.linalg.norm(up_vecs, axis=1, keepdims=True)
    up = up_vecs.mean(axis=0)
    up /= np.linalg.norm(up)
    poses = poses @ _rotation_to_z(up).T
    med = float(np.median(np.linalg.norm(poses[:, 5] - poses[:, 6], axis=1)))
    # Spec says [0.3, 0.5]; H3WB's COCO shoulder joints give a median of ~0.294 m, so the lower bound
    # is 0.25. The check still catches a wrong unit (mm would be ~294).
    assert 0.25 <= med <= 0.5, f"median shoulder width {med:.3f} m outside [0.25, 0.5]"
    return poses.astype(np.float32)


def _root(poses: np.ndarray) -> np.ndarray:
    return poses[:, list(ROOT_JOINTS)].mean(axis=1)


def sample_cameras(n: int, elev_range_deg: list, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """(n,7): azimuth_deg, elevation_deg, distance_m, focal_px, yaw_offset_deg, pitch_down_deg, roll_deg.

    Like a Tello: the camera is fixed to the body, so it is not aimed at the person. The drone heads roughly
    toward the person (yaw offset), the camera pitch is near level, and the person can sit anywhere in the
    frame or partly outside it. Azimuth/elevation/distance place the camera relative to the root (hips)."""
    d = cfg["data"]
    az = rng.uniform(0.0, 360.0, n)
    el = rng.uniform(elev_range_deg[0], elev_range_deg[1], n)
    dist = np.exp(rng.uniform(np.log(d["dist_m"][0]), np.log(d["dist_m"][1]), n))   # log-uniform: more close-ups
    focal = rng.uniform(d["focal_px"][0], d["focal_px"][1], n)
    yaw_off = rng.uniform(-d["yaw_offset_deg"], d["yaw_offset_deg"], n)
    pitch = rng.uniform(d["pitch_down_deg"][0], d["pitch_down_deg"][1], n)
    roll = rng.uniform(-d["roll_deg"], d["roll_deg"], n)
    return np.stack([az, el, dist, focal, yaw_off, pitch, roll], axis=1).astype(np.float32)


def camera_extrinsics(cams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cams (n,7) -> R (n,3,3) world->camera (rows: right, down, forward), pos (n,3) camera position in the
    root-relative, z-up world."""
    c = cams.astype(np.float64)
    az, el, dist = np.deg2rad(c[:, 0]), np.deg2rad(c[:, 1]), c[:, 2]
    pos = dist[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    heading = az + np.pi + np.deg2rad(c[:, 4])                 # toward the person, plus the yaw offset
    pitch, roll = np.deg2rad(c[:, 5]), np.deg2rad(c[:, 6])
    fwd = np.stack([np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading), -np.sin(pitch)], axis=1)
    right = np.stack([np.sin(heading), -np.cos(heading), np.zeros_like(heading)], axis=1)   # fwd x up, unit
    down = np.cross(fwd, right)
    r_c, r_s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    right, down = r_c * right + r_s * down, -r_s * right + r_c * down
    return np.stack([right, down, fwd], axis=1), pos


def normalize_2d(uv: np.ndarray, mask: np.ndarray, focal: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """uv (n,17,2) px, mask (n,17), focal (n,) -> kp2d (n,17,2), frame (n,3).

    kp2d: minus the 2D root (mean of visible hips; mean of visible joints if both hips are hidden), divided by
    the bounding-box diagonal of visible joints, hidden joints zeroed.
    frame: where the person is in the image, in focal units: ((root_u - W/2)/f, (root_v - H/2)/f, diag/f).
    Needed because the camera is not aimed at the person, so the viewing ray changes the 3D answer."""
    hip_mask = mask[:, list(ROOT_JOINTS)]
    hip_count = hip_mask.sum(axis=1)
    hip_root = (uv[:, list(ROOT_JOINTS)] * hip_mask[..., None]).sum(axis=1) / np.maximum(hip_count, 1)[:, None]
    vis_count = np.maximum(mask.sum(axis=1), 1)
    vis_mean = (uv * mask[..., None]).sum(axis=1) / vis_count[:, None]
    root2d = np.where((hip_count > 0)[:, None], hip_root, vis_mean)
    big = 1e9
    uv_min = np.where(mask[..., None] > 0, uv, big).min(axis=1)
    uv_max = np.where(mask[..., None] > 0, uv, -big).max(axis=1)
    diag = np.linalg.norm(uv_max - uv_min, axis=1)
    diag = np.where(np.isfinite(diag) & (diag > 1e-6), diag, 1.0)
    kp2d = (uv - root2d[:, None, :]) / diag[:, None, None] * mask[..., None]
    frame = np.stack([(root2d[:, 0] - W / 2) / focal, (root2d[:, 1] - H / 2) / focal, diag / focal], axis=1)
    return kp2d, frame


def project(poses: np.ndarray, cams: np.ndarray, rng: np.random.Generator, cfg: dict) -> dict:
    """poses (n,17,3) -> {'kp2d': (n,17,2) normalized, 'mask': (n,17), 'frame': (n,3),
    'pose3d': (n,17,3) camera frame, root-relative, 'cam': (n,7)}."""
    d = cfg["data"]
    W, H = d["sensor_px"]
    poses = poses.astype(np.float64)
    R, pos = camera_extrinsics(cams)
    focal = cams[:, 3].astype(np.float64)
    dist = cams[:, 2].astype(np.float64)

    cam_j = np.einsum("naj,nij->nia", R, poses - pos[:, None, :])   # cam_j[n,i,a] = R[n,a,:] . rel[n,i,:]
    pose3d = cam_j - _root(cam_j)[:, None, :]                       # camera frame, root-relative

    z = np.maximum(cam_j[..., 2], 1e-6)
    u = focal[:, None] * cam_j[..., 0] / z + W / 2
    v = focal[:, None] * cam_j[..., 1] / z + H / 2
    std = (d["noise_px_base"] + d["noise_px_per_m"] * dist)[:, None]
    u = u + rng.normal(0.0, 1.0, u.shape) * std
    v = v + rng.normal(0.0, 1.0, v.shape) * std
    uv = np.stack([u, v], axis=-1)

    # Random drops, plus anything outside the image or behind the camera (truncation).
    p_drop = np.full(17, d["drop_p"])
    p_drop[list(EXTREMITIES)] = d["drop_p_extremities"]
    dropped = rng.uniform(size=uv.shape[:2]) < p_drop[None, :]
    outside = (u < 0) | (u >= W) | (v < 0) | (v >= H) | (cam_j[..., 2] <= 0)
    mask = (~(dropped | outside)).astype(np.float64)

    kp2d, frame = normalize_2d(uv, mask, focal, W, H)
    return {
        "kp2d": kp2d.astype(np.float32),
        "mask": mask.astype(np.float32),
        "frame": frame.astype(np.float32),
        "pose3d": pose3d.astype(np.float32),
        "cam": cams.astype(np.float32),
        "truncated": outside.any(axis=1),
    }


def make_views(poses: np.ndarray, n_views: int, elev_range_deg: list, seed: int, cfg: dict) -> dict:
    """Repeat each pose n_views times with fresh cameras; same keys as project() minus 'truncated'.
    Views with fewer than data.min_visible_joints visible joints get a new camera (person mostly out of frame)."""
    rng = np.random.default_rng(seed)
    rep = np.repeat(poses.astype(np.float32), n_views, axis=0)
    out = project(rep, sample_cameras(rep.shape[0], elev_range_deg, rng, cfg), rng, cfg)
    for _ in range(100):
        bad = np.flatnonzero(out["mask"].sum(axis=1) < cfg["data"]["min_visible_joints"])
        if bad.size == 0:
            break
        redo = project(rep[bad], sample_cameras(bad.size, elev_range_deg, rng, cfg), rng, cfg)
        for k in out:
            out[k][bad] = redo[k]
    else:
        raise RuntimeError("could not place enough joints in frame; check data camera ranges")
    out["truncated_frac"] = float(out.pop("truncated").mean())
    return out
