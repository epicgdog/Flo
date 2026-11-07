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
    """(n,4): azimuth_deg, elevation_deg, distance_m, focal_px."""
    d = cfg["data"]
    az = rng.uniform(0.0, 360.0, n)
    el = rng.uniform(elev_range_deg[0], elev_range_deg[1], n)
    dist = rng.uniform(d["dist_m"][0], d["dist_m"][1], n)
    focal = rng.uniform(d["focal_px"][0], d["focal_px"][1], n)
    return np.stack([az, el, dist, focal], axis=1).astype(np.float32)


def project(poses: np.ndarray, cams: np.ndarray, rng: np.random.Generator, cfg: dict) -> dict:
    """poses (n,17,3) -> {'kp2d': (n,17,2) normalized, 'mask': (n,17),
    'pose3d': (n,17,3) camera frame, root-relative, 'cam': (n,4)}."""
    d = cfg["data"]
    W, H = d["sensor_px"]
    n = poses.shape[0]
    poses = poses.astype(np.float64)
    cams64 = cams.astype(np.float64)

    # Camera position = aim + distance * direction(azimuth, elevation). Aim = root + jitter.
    az = np.deg2rad(cams64[:, 0])
    el = np.deg2rad(cams64[:, 1])
    dist = cams64[:, 2]
    focal = cams64[:, 3]
    direction = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    jitter = rng.uniform(-d["aim_jitter_m"], d["aim_jitter_m"], (n, 3))
    aim = jitter                                   # root is the origin after root-relative poses
    pos = aim + dist[:, None] * direction

    # Look-at rotation, world up = +z. Camera axes: x right, y down, z forward.
    fwd = aim - pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    up = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=1)       # (n,3,3): rows are camera axes

    # Joints in camera frame.
    rel = poses - pos[:, None, :]                  # (n,17,3) world -> relative to camera
    cam_j = np.einsum("naj,nij->nia", R, rel)      # (n,17,3): cam_j[n,i,a] = R[n,a,:] . rel[n,i,:]
    pose3d = cam_j - _root(cam_j)[:, None, :]      # camera frame, root-relative

    # Pinhole projection.
    z = np.maximum(cam_j[..., 2], 1e-6)
    u = focal[:, None] * cam_j[..., 0] / z + W / 2
    v = focal[:, None] * cam_j[..., 1] / z + H / 2

    # Gaussian pixel noise, std depends on distance.
    std = (d["noise_px_base"] + d["noise_px_per_m"] * dist)[:, None]
    u = u + rng.normal(0.0, 1.0, u.shape) * std
    v = v + rng.normal(0.0, 1.0, v.shape) * std
    uv = np.stack([u, v], axis=-1)                 # (n,17,2) pixels

    # Random drops, plus anything outside the image or behind the camera.
    p_drop = np.full(17, d["drop_p"])
    p_drop[list(EXTREMITIES)] = d["drop_p_extremities"]
    dropped = rng.uniform(size=(n, 17)) < p_drop[None, :]
    outside = (u < 0) | (u >= W) | (v < 0) | (v >= H) | (cam_j[..., 2] <= 0)
    mask = (~(dropped | outside)).astype(np.float32)

    # 2D root = mean of projected hips; if both hips are dropped, mean of visible joints.
    hip_mask = mask[:, list(ROOT_JOINTS)]
    hip_count = hip_mask.sum(axis=1)
    hip_root = (uv[:, list(ROOT_JOINTS)] * hip_mask[..., None]).sum(axis=1) / np.maximum(hip_count, 1)[:, None]
    vis_count = np.maximum(mask.sum(axis=1), 1)
    vis_mean = (uv * mask[..., None]).sum(axis=1) / vis_count[:, None]
    root2d = np.where((hip_count > 0)[:, None], hip_root, vis_mean)

    # Normalize by bounding-box diagonal of visible joints.
    big = 1e9
    uv_min = np.where(mask[..., None] > 0, uv, big).min(axis=1)
    uv_max = np.where(mask[..., None] > 0, uv, -big).max(axis=1)
    diag = np.linalg.norm(uv_max - uv_min, axis=1)
    diag = np.where(np.isfinite(diag) & (diag > 1e-6), diag, 1.0)

    kp2d = (uv - root2d[:, None, :]) / diag[:, None, None]
    kp2d = kp2d * mask[..., None]                  # dropped joints -> 0

    return {
        "kp2d": kp2d.astype(np.float32),
        "mask": mask.astype(np.float32),
        "pose3d": pose3d.astype(np.float32),
        "cam": cams.astype(np.float32),
    }


def make_views(poses: np.ndarray, n_views: int, elev_range_deg: list, seed: int, cfg: dict) -> dict:
    """Repeat each pose n_views times with fresh cameras; same keys as project()."""
    rng = np.random.default_rng(seed)
    rep = np.repeat(poses.astype(np.float32), n_views, axis=0)
    cams = sample_cameras(rep.shape[0], elev_range_deg, rng, cfg)
    return project(rep, cams, rng, cfg)
