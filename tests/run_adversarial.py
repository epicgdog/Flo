"""Run the checker on each adversarial candidate and print PASS/FAIL against its expected status.

Run: uv run python3 tests/run_adversarial.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.check import check  # noqa: E402
from harness import config as C  # noqa: E402

ADV = os.path.join(ROOT, "tests", "adversarial")
OUT = os.path.join(ROOT, "runs", "adversarial")

EXPECTED = {
    "crash": "train_crash",
    "timeout": "timeout",
    "nan": "nan",
    "too_large": "too_large",
    "implausible": "implausible",
    "static_heldout": "static_fail",
    "static_requests": "static_fail",
    "fixed_pose": "ok",
}


def baseline_dev_score(cfg: dict) -> float:
    """Mean dev score of the baseline from history/baseline.json, else compute it now."""
    path = os.path.join(C.abspath(cfg["paths"]["history"]), "baseline.json")
    if os.path.exists(path):
        with open(path) as f:
            b = json.load(f)
        scores = [r["score"] for r in b.get("reports", []) if r.get("set") == "dev" and r.get("score") is not None]
        if scores:
            return sum(scores) / len(scores)
    rep = check(os.path.join(ROOT, "candidate"), "dev", 0, "adv_baseline", os.path.join(OUT, "baseline"), cfg)
    return rep["score"]


def main() -> int:
    cfg = C.load()
    base = None
    failures = 0
    for name, expected in EXPECTED.items():
        cdir = os.path.join(ADV, name)
        rep = check(cdir, "dev", 0, f"adv_{name}", os.path.join(OUT, name), cfg)
        got = rep["status"]
        if expected == "ok":
            if base is None:
                base = baseline_dev_score(cfg)
            score = rep["score"]
            passed = got == "ok" and score is not None and score >= 2.0 * base
            detail = f"score={score} need>=2x baseline {base:.3f}"
        else:
            passed = got == expected
            detail = f"status={got}"
        if not passed:
            failures += 1
        print(f"{'PASS' if passed else 'FAIL'} {name}: expected {expected}, {detail}")
        if rep.get("error") and not passed:
            print(f"     error: {str(rep['error'])[-300:]}")
    print(f"{len(EXPECTED) - failures}/{len(EXPECTED)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
