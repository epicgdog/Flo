"""Git gate. Only accept() can move main, and only on checker numbers.

REPO is the git repo the functions act on. Tests point it at a temp repo.
"""
import math
import re
import subprocess

from harness import config as C

REPO = C.ROOT


def _git(*args: str, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def start_iteration(iter_id: str) -> None:
    _git("checkout", "main")
    _git("checkout", "-b", f"iter/{iter_id}")


def commit_candidate(iter_id: str, message: str) -> None:
    _git("add", "candidate/train.py")
    _git("commit", "-m", f"{iter_id}: {message}")


def paired_threshold(n: int, cfg: dict) -> float:
    """Mean paired difference (challenger - incumbent, same seed) that counts as a real change with n seeds.
    Two independently trained models differ by about sqrt(2) * sigma_dev, so the mean of n paired differences
    has spread sqrt(2) * sigma_dev / sqrt(n). The threshold is paired_margin_sigmas times that spread."""
    g = cfg["gate"]
    return g["paired_margin_sigmas"] * math.sqrt(2) * g["sigma_dev"] / math.sqrt(n)


def classify_dev(diffs: list[float], cfg: dict) -> str:
    """diffs[i] = challenger dev score - incumbent dev score on seed i. Negative means the challenger is better.
    -> 'pass' (real gain), 'near_miss' (leaning better, keep for confirmation), 'inconclusive' (inside seed noise),
    or 'regression' (real loss)."""
    n = len(diffs)
    m = sum(diffs) / n
    t = paired_threshold(n, cfg)
    if m <= -t:
        return "pass"
    if m <= -t / 2:
        return "near_miss"
    if m < t:
        return "inconclusive"
    return "regression"


def passes_heldout(score: float, best_heldout: float, cfg: dict) -> bool:
    g = cfg["gate"]
    return score <= best_heldout + g["heldout_margin_sigmas"] * g["sigma_heldout"]


def _next_tag() -> str:
    tags = _git("tag", "--list", "v*", check=False).split()
    nums = [int(t[1:]) for t in tags if re.fullmatch(r"v\d+", t)]
    return f"v{(max(nums) + 1) if nums else 1}"


def accept(iter_id: str, dev: float, heldout: float) -> str:
    """Merge into main, tag, delete the iteration branch. Returns the tag."""
    tag = _next_tag()
    _git("checkout", "main")
    _git("merge", "--no-ff", f"iter/{iter_id}", "-m",
         f"accept {iter_id} dev={dev:.3f} heldout={heldout:.3f}")
    _git("tag", tag)
    _git("branch", "-d", f"iter/{iter_id}")
    return tag


def reject(iter_id: str) -> None:
    _git("checkout", "main")
    _git("branch", "-D", f"iter/{iter_id}")
