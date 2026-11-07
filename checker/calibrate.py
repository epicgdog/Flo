"""Calibrate: baseline x 4 seeds x {dev, heldout}. Writes sigma into config.yaml, history/baseline.json,
the v0 baseline (history/0000/outcome.json, history/0000/model/) and git tag v0.

Run: uv run python3 checker/calibrate.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.check import check  # noqa: E402
from harness import config as C  # noqa: E402

SEEDS = [0, 1, 2, 3]


def _set_value(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^(\s*{key}:\s*)[^\s#]+", lambda m: m.group(1) + value, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"no '{key}:' line in config.yaml")
    return new


def main() -> None:
    """Baseline x 4 seeds x {dev, heldout}. Write sigma_dev, sigma_heldout to config.yaml;
    write history/baseline.json with all 8 reports and the mean scores."""
    cfg = C.load()
    hist = C.abspath(cfg["paths"]["history"])
    cand = C.abspath(os.path.dirname(cfg["paths"]["candidate"]))
    if subprocess.run(["git", "status", "--porcelain", cfg["paths"]["candidate"]], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        raise SystemExit("candidate/train.py has uncommitted changes; commit the baseline before tagging v0")

    reports = []
    for set_name in ("dev", "heldout"):
        for seed in SEEDS:
            out = os.path.join(ROOT, "runs", "work", f"0000_{set_name}_{seed}")
            rep = check(cand, set_name, seed, "0000", out, cfg)
            print(f"{set_name} seed {seed}: status={rep['status']} score={rep['score']}", flush=True)
            reports.append(rep)
    bad = [(r["set"], r["seed"], r["status"]) for r in reports if r["status"] != "ok"]
    if bad:
        raise SystemExit(f"baseline runs not ok: {bad}")

    dev = np.array([r["score"] for r in reports if r["set"] == "dev"])
    ho = np.array([r["score"] for r in reports if r["set"] == "heldout"])
    sigma_dev, sigma_ho = float(dev.std(ddof=1)), float(ho.std(ddof=1))
    os.makedirs(os.path.join(hist, "0000"), exist_ok=True)
    with open(os.path.join(hist, "baseline.json"), "w") as f:
        json.dump({"reports": reports, "mean_dev": float(dev.mean()), "mean_heldout": float(ho.mean()),
                   "sigma_dev": sigma_dev, "sigma_heldout": sigma_ho}, f, indent=2)
    print(f"dev: mean {dev.mean():.3f}, sigma {sigma_dev:.3f} ({100 * sigma_dev / dev.mean():.1f}% of mean)")
    print(f"heldout: mean {ho.mean():.3f}, sigma {sigma_ho:.3f}")
    if sigma_dev > 0.05 * dev.mean():
        raise SystemExit("STOP: sigma_dev is over 5% of the baseline dev score. Tell the person; nothing else written.")

    text = open(os.path.join(ROOT, "config.yaml")).read()
    text = _set_value(text, "sigma_dev", f"{sigma_dev:.4f}")
    text = _set_value(text, "sigma_heldout", f"{sigma_ho:.4f}")
    open(os.path.join(ROOT, "config.yaml"), "w").write(text)

    # v0: the gate compares candidates against the baseline's seed-0 dev and held-out scores.
    d0 = next(r for r in reports if r["set"] == "dev" and r["seed"] == 0)
    d1 = next(r for r in reports if r["set"] == "dev" and r["seed"] == 1)
    h0 = next(r for r in reports if r["set"] == "heldout" and r["seed"] == 0)
    outcome = {"iter_id": "0000", "strategy": None, "rationale": "baseline train.py", "accepted": True,
               "reason": None, "tag": "v0", "dev": d0["score"], "dev2": d1["score"], "heldout": h0["score"],
               "best_dev_before": None, "best_heldout_before": None, "triage": None, "lesson": None,
               "duration_s": None, "llm_tokens": 0}
    with open(os.path.join(hist, "0000", "outcome.json"), "w") as f:
        json.dump(outcome, f, indent=2)
    mdir = os.path.join(hist, "0000", "model")
    os.makedirs(mdir, exist_ok=True)
    for fn in ("model.pt", "model_def.py"):
        shutil.copy(os.path.join(ROOT, "runs", "work", "0000_dev_0", fn), os.path.join(mdir, fn))
    shutil.copy(os.path.join(ROOT, "runs", "work", "0000_dev_0", "train_log.jsonl"),
                os.path.join(hist, "0000", "train_log_dev.jsonl"))       # dashboard: v0 loss curve
    shutil.rmtree(os.path.join(ROOT, "runs", "work"), ignore_errors=True)
    subprocess.run(["git", "tag", "-f", "v0"], cwd=ROOT, check=True)
    print(f"wrote sigma_dev={sigma_dev:.4f}, sigma_heldout={sigma_ho:.4f}; history/0000; tagged v0")


if __name__ == "__main__":
    main()
