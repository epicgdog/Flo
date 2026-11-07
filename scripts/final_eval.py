"""M14: score v0 and the latest tag with 3 seeds on dev and held-out; write history/final.json
with means and standard deviations. Each tag's train.py is read from git and run through the checker.

Run: uv run python3 scripts/final_eval.py
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

SEEDS = [0, 1, 2]


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def latest_tag() -> str:
    tags = [t for t in _git("tag", "--list", "v*").split() if re.fullmatch(r"v\d+", t)]
    return max(tags, key=lambda t: int(t[1:]))


def main() -> None:
    cfg = C.load()
    work = os.path.join(ROOT, "runs", "work", "final")
    result = {}
    for tag in dict.fromkeys(["v0", latest_tag()]):
        cand = os.path.join(work, tag)
        os.makedirs(cand, exist_ok=True)
        with open(os.path.join(cand, "train.py"), "w") as f:
            f.write(_git("show", f"{tag}:{cfg['paths']['candidate']}"))
        result[tag] = {}
        for set_name in ("dev", "heldout"):
            reps = [check(cand, set_name, s, f"final_{tag}", os.path.join(cand, f"{set_name}_{s}"), cfg) for s in SEEDS]
            scores = [r["score"] for r in reps if r["score"] is not None]
            result[tag][set_name] = {"scores": [r["score"] for r in reps], "statuses": [r["status"] for r in reps],
                                     "mean": float(np.mean(scores)) if scores else None,
                                     "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None}
            print(f"{tag} {set_name}: mean {result[tag][set_name]['mean']} std {result[tag][set_name]['std']}", flush=True)
    out = os.path.join(C.abspath(cfg["paths"]["history"]), "final.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
