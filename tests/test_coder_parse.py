import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.coder import apply_candidate, parse_candidates  # noqa: E402

SAMPLE = """```xml
<candidate id="A">
<rationale>Lower learning rate for stability.</rationale>
<edit><search>lr=1e-4</search><replace>lr=5e-5</replace></edit>
</candidate>
<candidate id="B">
<rationale>No edits here.</rationale>
</candidate>
<candidate id="C">
<rationale>Double the batch.</rationale>
<edit><search>BATCH = 256</search><replace>BATCH = 512</replace></edit>
</candidate>
```"""

TRAIN = "BATCH = 256\nopt = AdamW(lr=1e-4)\n"


def test_fences_and_zero_edit_dropped():
    cands = parse_candidates(SAMPLE)
    assert [c.id for c in cands] == ["A", "C"]
    assert cands[0].rationale.startswith("Lower learning rate")
    assert cands[0].edits == [("lr=1e-4", "lr=5e-5")]


def test_apply_exact_once():
    cand = parse_candidates(SAMPLE)[1]
    out = apply_candidate(TRAIN, cand)
    assert out is not None and "BATCH = 512" in out


def test_apply_rejects_missing_or_ambiguous():
    cand = parse_candidates(SAMPLE)[0]
    assert apply_candidate("opt = 1\n", cand) is None          # not found
    assert apply_candidate("lr=1e-4\nlr=1e-4\n", cand) is None  # found twice


def test_apply_rejects_bad_syntax():
    bad = parse_candidates(SAMPLE.replace("BATCH = 512", "BATCH = (512"))[1]
    assert apply_candidate(TRAIN, bad) is None
