"""Checker: static check, train in a subprocess, load, sample, aggregate, sanity, metrics, score, plot.

CLI: uv run python3 checker/check.py <candidate_dir> --set dev|heldout --seed 0 --out <dir>
"""
import argparse
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.datagen import BONES  # noqa: E402
from checker.metrics import best_of_k_mm, bone_err, mpjpe_mm, pa_mpjpe_mm, score, spread_mm  # noqa: E402
from checker.sampler import sample_hypotheses  # noqa: E402
from harness import config as C  # noqa: E402

FORBIDDEN_IMPORTS = {"requests", "urllib", "socket", "http", "subprocess"}
FORBIDDEN_STRINGS = ("eval.npz", "heldout", "checker")


def static_check(train_py: str) -> tuple[bool, str]:
    try:
        with open(train_py) as f:
            src = f.read()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as e:
        return False, f"ast.parse failed: {e}"
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.split(".")[0] in FORBIDDEN_IMPORTS:
                return False, f"forbidden import {n}"
    for s in FORBIDDEN_STRINGS:
        if s in src:
            return False, f"forbidden string {s!r}"
    return True, "ok"


def _read_log(out_dir: str) -> list[dict]:
    p = os.path.join(out_dir, "train_log.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def run_training(train_py: str, out_dir: str, seed: int, cfg: dict,
                 stop_event: threading.Event | None = None) -> dict:
    """-> {status, elapsed_s, steps, final_loss, killed_early, stderr_tail}"""
    tr = cfg["train"]
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="check_train_")
    try:
        shutil.copy(train_py, os.path.join(tmp, "train.py"))
        cmd = [sys.executable, os.path.join(tmp, "train.py"), "--data", C.abspath(cfg["paths"]["train"]),
               "--out", out_dir, "--time-budget", str(tr["time_budget_s"]), "--seed", str(seed)]
        t0 = time.time()
        err_path = os.path.join(tmp, "stderr.txt")
        with open(err_path, "w") as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, cwd=tmp)
            while True:
                rc = proc.poll()
                if rc is not None:
                    status = "ok" if rc == 0 else "train_crash"
                    break
                if stop_event is not None and stop_event.is_set():
                    proc.kill()
                    proc.wait()
                    status = "killed_early"
                    break
                if time.time() - t0 > tr["kill_after_s"]:
                    proc.kill()
                    proc.wait()
                    status = "timeout"
                    break
                time.sleep(0.2)
        elapsed = time.time() - t0
        with open(err_path) as f:
            stderr_tail = f.read()[-2000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log = _read_log(out_dir)
    return {"status": status, "elapsed_s": round(elapsed, 2),
            "steps": log[-1].get("step") if log else None,
            "final_loss": log[-1].get("loss") if log else None,
            "killed_early": status == "killed_early", "stderr_tail": stderr_tail}


def load_model(model_dir: str):
    """Import model_def.py from model_dir and load model.pt. -> (model, n_params)."""
    spec = importlib.util.spec_from_file_location(f"model_def_{abs(hash(model_dir))}",
                                                  os.path.join(model_dir, "model_def.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load(os.path.join(model_dir, "model.pt"))
    model.eval()
    return model, sum(p.numel() for p in model.parameters())


def predict(model, kp2d: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """kp2d (N,17,2), mask (N,17) -> hypotheses (N,K,17,3), batches of sample.batch, seed sample.noise_seed."""
    s = cfg["sample"]
    cond = torch.from_numpy(np.concatenate([kp2d.reshape(-1, 34), mask.reshape(-1, 17)], axis=1).astype(np.float32))
    N = cond.shape[0]
    out = np.zeros((N, s["k_hypotheses"], 17, 3), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, N, s["batch"]):
            h = sample_hypotheses(model, cond[i:i + s["batch"]], s["k_hypotheses"], s["euler_steps"], s["noise_seed"])
            out[i:i + s["batch"]] = h.reshape(-1, s["k_hypotheses"], 17, 3).double().numpy()
    return out


def evaluate(out_dir: str, eval_npz: str, cfg: dict) -> dict:
    """-> {status, metrics, score} (plus error, params); also writes skeletons.png into out_dir."""
    sc = cfg["score"]
    res = {"status": None, "metrics": None, "score": None, "error": None, "params": None}
    try:
        model, n_params = load_model(out_dir)
    except Exception as e:
        return {**res, "status": "load_fail", "error": repr(e)}
    res["params"] = n_params
    if n_params > cfg["train"]["max_params"]:
        return {**res, "status": "too_large", "error": f"{n_params} params > {cfg['train']['max_params']}"}

    d = np.load(eval_npz)
    gt = d["pose3d"].astype(np.float64)
    try:
        hyps = predict(model, d["kp2d"], d["mask"], cfg)
    except Exception as e:
        return {**res, "status": "nan", "error": f"sampling failed: {e!r}"}
    if not np.all(np.isfinite(hyps)):
        return {**res, "status": "nan", "error": "NaN or inf in prediction"}
    pred = hyps.mean(axis=1)
    root = pred[:, [11, 12]].mean(axis=1, keepdims=True)
    if np.linalg.norm(pred - root, axis=-1).max() > sc["max_joint_dist_m"]:
        return {**res, "status": "nan", "error": f"joint farther than {sc['max_joint_dist_m']} m from the root"}

    metrics = {"mpjpe_mm": mpjpe_mm(pred, gt), "pa_mpjpe_mm": pa_mpjpe_mm(pred, gt),
               "best_of_k_mm": best_of_k_mm(hyps, gt), "spread_mm": spread_mm(hyps), "bone_err": bone_err(pred, gt)}
    res["metrics"] = metrics
    if metrics["bone_err"] > sc["implausible_bone_err"]:
        return {**res, "status": "implausible",
                "error": f"bone_err {metrics['bone_err']:.3f} > {sc['implausible_bone_err']}"}
    _plot_skeletons(os.path.join(out_dir, "skeletons.png"), gt, pred, hyps)
    return {**res, "status": "ok", "score": score(metrics, cfg)}


def _plot_skeletons(path: str, gt: np.ndarray, pred: np.ndarray, hyps: np.ndarray) -> None:
    n = min(3, gt.shape[0])
    fig = plt.figure(figsize=(12, 4))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for a, b in BONES:
            for h in hyps[i]:
                ax.plot([h[a, 0], h[b, 0]], [h[a, 2], h[b, 2]], [-h[a, 1], -h[b, 1]], color="tab:orange", lw=0.4, alpha=0.5)
            g, p = gt[i], pred[i]
            ax.plot([g[a, 0], g[b, 0]], [g[a, 2], g[b, 2]], [-g[a, 1], -g[b, 1]], color="grey", lw=2)
            ax.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], [-p[a, 1], -p[b, 1]], color="tab:blue", lw=2)
        ax.set_title(f"eval input {i}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def check(candidate_dir: str, set_name: str, seed: int, iter_id: str, out_dir: str,
          cfg: dict, stop_event: threading.Event | None = None) -> dict:
    """-> full report (section 8). Stops at the first failing step."""
    os.makedirs(out_dir, exist_ok=True)
    train_py = os.path.join(candidate_dir, "train.py")
    report = {"iter_id": iter_id, "set": set_name, "seed": seed, "status": None, "score": None, "error": None,
              "metrics": None, "train": None, "params": None, "skeletons_png": None}
    ok, why = static_check(train_py)
    if not ok:
        report.update(status="static_fail", error=why)
        return report
    tr = run_training(train_py, out_dir, seed, cfg, stop_event)
    report["train"] = {k: tr[k] for k in ("steps", "final_loss", "elapsed_s", "killed_early")}
    if tr["status"] != "ok":
        report.update(status=tr["status"], error=tr["stderr_tail"].strip() or tr["status"])
        return report
    ev = evaluate(out_dir, C.abspath(cfg["paths"]["dev" if set_name == "dev" else "heldout"]), cfg)
    report.update(status=ev["status"], score=ev["score"], metrics=ev["metrics"], error=ev["error"], params=ev["params"])
    png = os.path.join(out_dir, "skeletons.png")
    if os.path.exists(png):
        report["skeletons_png"] = os.path.relpath(png, ROOT) if png.startswith(ROOT) else png
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_dir")
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = C.load()
    out = os.path.abspath(args.out)
    rep = check(os.path.abspath(args.candidate_dir), args.set, args.seed, "manual", out, cfg)
    with open(os.path.join(out, f"report_{args.set}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
