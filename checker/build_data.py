"""Build train.npz, dev_eval.npz, heldout_eval.npz and preview.png from the H3WB files.

Run: uv run python3 checker/build_data.py
"""
import hashlib
import io
import os
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker.datagen import BONES, load_h3wb_body, make_views  # noqa: E402
from harness import config as C  # noqa: E402

ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed timestamp so identical data gives identical bytes


def save_npz_deterministic(path: str, arrays: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE), buf.getvalue())


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_preview(path: str, train: dict, cfg: dict) -> None:
    rng = np.random.default_rng(cfg["data"]["seeds"]["train"])
    idx = rng.choice(train["kp2d"].shape[0], 6, replace=False)
    fig = plt.figure(figsize=(8, 18))
    for row, i in enumerate(idx):
        ax2 = fig.add_subplot(6, 2, 2 * row + 1)
        kp = train["kp2d"][i]
        m = train["mask"][i] > 0
        ax2.scatter(kp[m, 0], -kp[m, 1], s=8)
        for a, b in BONES:
            if m[a] and m[b]:
                ax2.plot([kp[a, 0], kp[b, 0]], [-kp[a, 1], -kp[b, 1]], lw=1)
        ax2.set_title(f"kp2d sample {i}", fontsize=8)
        ax2.set_aspect("equal")
        ax3 = fig.add_subplot(6, 2, 2 * row + 2, projection="3d")
        p = train["pose3d"][i]
        # camera frame (x right, y down, z forward) drawn with image-up as the plot's up axis
        ax3.scatter(p[:, 0], p[:, 2], -p[:, 1], s=6)
        for a, b in BONES:
            ax3.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], [-p[a, 1], -p[b, 1]], lw=1)
        ax3.set_title("pose3d (camera frame)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(config_path: str = "config.yaml") -> None:
    """Write train.npz, dev_eval.npz, heldout_eval.npz and preview.png. Print shapes,
    mean shoulder width (m), and the sha256 of each npz."""
    cfg = C.load(C.abspath(config_path))
    p = cfg["paths"]
    d = cfg["data"]
    missing = [q for q in (p["raw_train_json"], p["raw_test_json"]) if not os.path.exists(C.abspath(q))]
    if missing:
        raise SystemExit(f"missing H3WB files: {missing}. Put them in data/raw/ and set paths.raw_test_json.")

    train_all = load_h3wb_body(C.abspath(p["raw_train_json"]), cfg)
    test_all = load_h3wb_body(C.abspath(p["raw_test_json"]), cfg)
    n_train = d["train_poses"]
    train_poses = train_all[:n_train]
    dev_pool = train_all[n_train:]
    n_eval = d["eval_samples"]
    seeds = d["seeds"]

    dev_idx = np.random.default_rng(seeds["dev"]).choice(len(dev_pool), min(n_eval, len(dev_pool)), replace=False)
    ho_idx = np.random.default_rng(seeds["heldout"]).choice(len(test_all), min(n_eval, len(test_all)), replace=False)

    splits = {
        p["train"]: make_views(train_poses, d["views_per_pose"], d["elev_train_deg"], seeds["train"], cfg),
        p["dev"]: make_views(dev_pool[dev_idx], 1, d["elev_train_deg"], seeds["dev"], cfg),
        p["heldout"]: make_views(test_all[ho_idx], 1, d["elev_heldout_deg"], seeds["heldout"], cfg),
    }
    os.makedirs(os.path.dirname(C.abspath(p["train"])), exist_ok=True)
    for rel, arrays in splits.items():
        save_npz_deterministic(C.abspath(rel), arrays)
        print(f"{rel}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in arrays.items()))

    allp = np.concatenate([train_all, test_all])
    sw = np.linalg.norm(allp[:, 5] - allp[:, 6], axis=1)
    print(f"raw poses: train json {len(train_all)}, test {len(test_all)}")
    print(f"mean shoulder width (m): {sw.mean():.4f}")
    print(f"median shoulder width (m): {np.median(sw):.4f}")
    for rel in splits:
        print(f"sha256 {rel}: {sha256(C.abspath(rel))}")
    preview = os.path.join(os.path.dirname(C.abspath(p["train"])), "preview.png")
    write_preview(preview, splits[p["train"]], cfg)
    print(f"preview: {preview}")


if __name__ == "__main__":
    main()
