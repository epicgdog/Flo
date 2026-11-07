"""Git gate. Only accept() can move main, and only on checker numbers.

REPO is the git repo the functions act on. Tests point it at a temp repo.
"""
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


def passes_dev(dev_scores: list[float], best_dev: float, cfg: dict) -> bool:
    """Mean of dev_scores must beat best_dev by dev_margin_sigmas * sigma_dev.
    Rule 1 passes [seed0]; rule 2 passes [seed0, seed1]."""
    g = cfg["gate"]
    margin = g["dev_margin_sigmas"] * g["sigma_dev"]
    return sum(dev_scores) / len(dev_scores) < best_dev - margin


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
