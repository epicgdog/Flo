import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import config as C  # noqa: E402
import loop  # noqa: E402

CFG = C.load()
BEST = {"tag": "v1", "iter_id": "0002", "dev": 70.0, "dev2": 71.0, "heldout": 110.0}


def _fake(monkeypatch, dev_scores, heldout=105.0, inc2=None, statuses=None):
    """dev_scores: scores returned by successive training runs. statuses: optional status per label."""
    calls = iter(dev_scores)
    seen = []

    def train_stage(cfg, iter_id, set_name, seed, label, phase, context):
        seen.append(label)
        st = (statuses or {}).get(label, "ok")
        return {"status": st, "score": next(calls) if st == "ok" else None, "error": "boom" if st != "ok" else None}

    def evaluate(out_dir, npz, cfg):
        return {"status": "ok", "score": heldout, "metrics": {}, "error": None, "params": 1}

    monkeypatch.setattr(loop, "train_stage", train_stage)
    monkeypatch.setattr(loop, "evaluate", evaluate)
    monkeypatch.setattr(loop.history, "save", lambda *a, **k: None)
    if inc2 is not None:
        monkeypatch.setattr(loop, "incumbent_score", lambda cfg, best, seed: inc2)
    return seen


def test_pass_uses_one_checkpoint_for_both_splits_and_two_seeds(monkeypatch):
    seen = _fake(monkeypatch, [68.0, 69.5])            # diffs -2.0, -1.5 -> mean -1.75 <= -1.0
    out = {}
    cls, reason, _ = loop.full_eval(CFG, "0030", "ctx", BEST, out)
    assert (cls, reason) == ("pass", None)
    assert seen == ["dev", "dev2"]                      # no separate held-out training run
    assert out["heldout"] == 105.0


def test_near_miss_goes_to_third_seed_with_incumbent_run(monkeypatch):
    seen = _fake(monkeypatch, [69.5, 70.4, 69.4], inc2=70.9)   # diffs -0.5, -0.6 -> mean -0.55 (near miss); dev3 diff -1.5
    out = {}
    cls, reason, detail = loop.full_eval(CFG, "0030", "ctx", BEST, out)
    assert seen == ["dev", "dev2", "dev3"] and out["dev3"] == 69.4
    assert cls == "pass" and reason is None, detail


def test_inconclusive_is_not_a_regression(monkeypatch):
    _fake(monkeypatch, [69.6, 70.9])                    # diffs -0.4, -0.1 -> mean -0.25: inside noise
    cls, reason, _ = loop.full_eval(CFG, "0030", "ctx", BEST, {})
    assert (cls, reason) == ("inconclusive", "no_gain")


def test_heldout_regression_blocks_pass(monkeypatch):
    _fake(monkeypatch, [68.0, 69.5], heldout=130.0)
    cls, reason, _ = loop.full_eval(CFG, "0030", "ctx", BEST, {})
    assert (cls, reason) == ("heldout_regression", "heldout_regression")


def test_seed0_regression_stops_before_seed1(monkeypatch):
    seen = _fake(monkeypatch, [75.0])
    cls, reason, _ = loop.full_eval(CFG, "0030", "ctx", BEST, {})
    assert (cls, reason) == ("regression", "no_gain")
    assert seen == ["dev"]


def test_crash_is_classified_as_crash(monkeypatch):
    _fake(monkeypatch, [], statuses={"dev": "train_crash"})
    cls, reason, detail = loop.full_eval(CFG, "0030", "ctx", BEST, {})
    assert (cls, reason) == ("crash", "train_crash") and detail == "boom"


def test_append_lesson_routes_by_class(tmp_path, monkeypatch):
    skills = tmp_path / "skills.md"
    skills.write_text("## Confirmed (passed held-out)\n- a\n\n## Failed ideas (do not retry)\n- b\n")
    cfg = {"paths": {"skills": str(skills)}}
    monkeypatch.setattr(loop.C, "abspath", lambda p: str(skills))
    loop.append_lesson(cfg, "crash", "- [0031] crashed")
    loop.append_lesson(cfg, "inconclusive", "- [0032] within noise")
    text = skills.read_text()
    assert text.index("- [0031]") > text.index("## Untested")
    assert text.index("- [0032]") > text.index("## Inconclusive")
