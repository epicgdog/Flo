import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import config as C  # noqa: E402
from harness import events, gate  # noqa: E402


def _run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    d = str(tmp_path)
    _run(d, "init", "-q", "-b", "main")
    _run(d, "config", "user.email", "test@example.com")
    _run(d, "config", "user.name", "test")
    os.makedirs(os.path.join(d, "candidate"))
    with open(os.path.join(d, "candidate", "train.py"), "w") as f:
        f.write("BASE = 1\n")
    _run(d, "add", "-A")
    _run(d, "commit", "-q", "-m", "init")
    _run(d, "tag", "v0")
    monkeypatch.setattr(gate, "REPO", d)
    return d


def _propose(repo, iter_id, text):
    gate.start_iteration(iter_id)
    with open(os.path.join(repo, "candidate", "train.py"), "w") as f:
        f.write(text)
    gate.commit_candidate(iter_id, "test change")


def test_passing_candidate_merges_and_tags_v1(repo):
    _propose(repo, "0001", "BASE = 2\n")
    tag = gate.accept("0001", dev=10.0, heldout=12.0)
    assert tag == "v1"
    assert _run(repo, "branch", "--show-current") == "main"
    assert "BASE = 2" in open(os.path.join(repo, "candidate", "train.py")).read()
    assert "v1" in _run(repo, "tag", "--list").split()
    assert "iter/0001" not in _run(repo, "branch", "--list")


def test_failing_candidate_leaves_main_and_deletes_branch(repo):
    _propose(repo, "0002", "BASE = 3\n")
    gate.reject("0002")
    assert _run(repo, "branch", "--show-current") == "main"
    assert "BASE = 1" in open(os.path.join(repo, "candidate", "train.py")).read()
    assert "iter/0002" not in _run(repo, "branch", "--list")


def test_events_and_state_write_valid_json(tmp_path):
    import json
    cfg = C.load()
    cfg = {**cfg, "paths": {**cfg["paths"], "events": str(tmp_path / "events.jsonl"),
                            "state": str(tmp_path / "state.json")}}
    events.emit("iter_start", cfg, iter_id="0001")
    events.emit("accepted", cfg, iter_id="0001", tag="v1")
    lines = open(tmp_path / "events.jsonl").read().splitlines()
    recs = [json.loads(x) for x in lines]
    assert [r["type"] for r in recs] == ["iter_start", "accepted"]
    assert all("ts" in r and r["iter_id"] == "0001" for r in recs)
    events.set_state(cfg, phase="training_dev")
    events.set_state(cfg, train_loss=0.5)
    st = json.load(open(tmp_path / "state.json"))
    assert st["phase"] == "training_dev" and st["train_loss"] == 0.5 and "last_update" in st


def test_threshold_rules():
    cfg = {"gate": {"paired_margin_sigmas": 1.0, "heldout_margin_sigmas": 1.0, "sigma_dev": 0.7073, "sigma_heldout": 2.0}}
    assert gate.passes_heldout(11.9, 10.0, cfg)
    assert not gate.passes_heldout(12.1, 10.0, cfg)


def test_paired_classes_with_two_seeds():
    cfg = {"gate": {"paired_margin_sigmas": 1.0, "sigma_dev": 0.7073}}
    # threshold for n=2 is sigma*sqrt(2)/sqrt(2) = 0.7073 mm
    assert gate.classify_dev([-1.46, -1.05], cfg) == "pass"        # the attempt-0024 pattern: mean -1.25
    assert gate.classify_dev([-0.5, -0.5], cfg) == "near_miss"     # mean -0.5, between -T/2 and -T
    assert gate.classify_dev([-0.1, 0.2], cfg) == "inconclusive"
    assert gate.classify_dev([1.5, 1.0], cfg) == "regression"


def test_single_seed_screen_uses_full_threshold():
    cfg = {"gate": {"paired_margin_sigmas": 1.0, "sigma_dev": 0.7073}}
    assert gate.paired_threshold(1, cfg) == pytest.approx(1.0, abs=1e-3)
    assert gate.classify_dev([-1.2], cfg) == "pass"
    assert gate.classify_dev([0.3], cfg) == "inconclusive"
