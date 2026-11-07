"""Coder agent: asks the Meta model for candidate edits to candidate/train.py, parses and applies them.

CLI: uv run python3 -m agents.coder --dry-run --strategy tune_optimizer
"""
import argparse
import ast
import difflib
import re
from dataclasses import dataclass

from agents.llm import chat
from harness import config as C
from harness import history

SYSTEM = """You improve a PyTorch training script for conditional flow matching.
The model maps 2D body keypoints (17 joints, normalized, plus the person's
position and size in the image) seen from a drone to a 3D pose (17 joints, meters, camera frame, root-relative).
Lower 3D error on unseen data is better.
Rules:
- Keep the command-line interface and output files exactly as they are.
- Exit within --time-budget seconds. At most 5,000,000 parameters.
- Read only the --data file. No network. No subprocess.
- Each candidate changes one idea. Keep edits small.
Return exactly 4 candidates in this format and nothing else:
<candidate id="A">
<rationale>one sentence</rationale>
<edit>
<search>exact lines from the current file</search>
<replace>new lines</replace>
</edit>
</candidate>
A candidate may contain several <edit> blocks."""

STRATEGIES = {
    "tune_optimizer": "Change optimizer, learning rate, schedule, batch size or EMA.",
    "timestep_sampling": "Change how t is sampled during training.",
    "architecture": "Change the network structure within the parameter limit.",
    "conditioning": "Change how the 2D keypoints and mask are encoded and fed in.",
    "loss_and_target": "Change the loss, its weighting, or what the network predicts.",
    "regularization": "Add dropout, augmentation of 2D inputs, or weight decay.",
    "fix_last_failure": "Fix the error in the last attempt.",
}

# Spec says max 100 tokens, but muse-spark spends ~85-98 output tokens on reasoning even at effort "minimal",
# which leaves an empty answer. 400 leaves room; the line itself is still cut to 200 characters.
LESSON_MAX_TOKENS = 400

LESSON_PROMPT = """Write one line (max 200 characters) stating what was tried and what happened,
as a reusable lesson. Attempt: {rationale}. Strategy: {strategy}.
Outcome: {outcome}. Dev change: {delta}."""


@dataclass
class Candidate:
    id: str                          # "A".. "D"
    rationale: str
    edits: list[tuple[str, str]]     # (search, replace)


_CAND_RE = re.compile(r'<candidate\s+id="([^"]*)"\s*>(.*?)</candidate>', re.S)
_RAT_RE = re.compile(r"<rationale>(.*?)</rationale>", re.S)
_EDIT_RE = re.compile(r"<edit>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</edit>", re.S)


def build_prompt(strategy: str, train_py: str, skills_md: str,
                 recent: list[dict], last_error: str | None) -> tuple[str, str]:
    """(system, user). User prompt order: STRATEGY, CURRENT train.py, SKILLS, RECENT ATTEMPTS, [LAST ERROR]."""
    lines = [f"STRATEGY: {strategy}: {STRATEGIES[strategy]}", "", "CURRENT train.py:", train_py, "",
             "SKILLS:", skills_md, "", "RECENT ATTEMPTS:"]
    for s in recent:
        lines.append(f"{s['id']} | {s['strategy']} | {s['rationale']} | dev={s['dev']} | {s['result']}")
    if strategy == "fix_last_failure":
        lines += ["", "LAST ERROR:", str(last_error)]
    return SYSTEM, "\n".join(lines)


def _strip_fences(text: str) -> str:
    return re.sub(r"^\s*```[A-Za-z]*\s*$", "", text, flags=re.M)


def parse_candidates(text: str) -> list[Candidate]:
    out = []
    for cid, body in _CAND_RE.findall(_strip_fences(text)):
        edits = _EDIT_RE.findall(body)
        if not edits:
            continue                       # zero edits: dropped
        rat = _RAT_RE.search(body)
        out.append(Candidate(cid.strip(), rat.group(1).strip() if rat else "", [(s, r) for s, r in edits]))
    return out


def _apply(train_py: str, cand: Candidate) -> tuple[str | None, str]:
    src = train_py
    for i, (search, replace) in enumerate(cand.edits):
        n = src.count(search) if search else 0
        if n != 1:
            return None, f"edit {i + 1}: search text found {n} times, must be exactly once"
        src = src.replace(search, replace, 1)
    try:
        ast.parse(src)
    except SyntaxError as e:
        return None, f"ast.parse failed: {e}"
    return src, "ok"


def apply_candidate(train_py: str, cand: Candidate) -> str | None:
    """None if any search string is not found exactly once, or ast.parse fails."""
    return _apply(train_py, cand)[0]


def propose(strategy: str, ctx: dict, cfg: dict) -> list[tuple[Candidate, str]]:
    """Valid (candidate, new_train_py) pairs; re-prompts once with errors if none are valid.
    ctx: train_py, skills_md, recent, last_error. Adds ctx['candidates_log'] and ctx['raw_responses']."""
    system, user = build_prompt(strategy, ctx["train_py"], ctx["skills_md"], ctx["recent"], ctx.get("last_error"))
    log, raws, valid = [], [], []
    for attempt in (1, 2):
        raw = chat(cfg["meta"]["coder_model"], system, user, cfg, effort=cfg["meta"]["coder_effort"])
        raws.append(raw)
        errors = []
        cands = parse_candidates(raw)
        if not cands:
            errors.append("no candidate with at least one <edit> block was found")
        for c in cands:
            new, why = _apply(ctx["train_py"], c)
            log.append({"attempt": attempt, "id": c.id, "rationale": c.rationale, "edits": c.edits,
                        "valid": new is not None, "error": None if new else why})
            if new is not None:
                valid.append((c, new))
            else:
                errors.append(f"candidate {c.id}: {why}")
        if valid:
            break
        user = user + "\n\nPREVIOUS ERRORS:\n" + "\n".join(errors)
    ctx["candidates_log"] = log
    ctx["raw_responses"] = raws
    return valid


def diff_text(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/train.py", "b/train.py"))


def write_lesson(iter_id: str, strategy: str, rationale: str, outcome: str,
                 dev_delta_mm: float | None, cfg: dict) -> str:
    """One line, max 200 chars. outcome is 'accepted' or 'rejected: <reason>'."""
    delta = "n/a" if dev_delta_mm is None else f"{dev_delta_mm:+.1f} mm"
    prompt = LESSON_PROMPT.format(rationale=rationale, strategy=strategy, outcome=outcome, delta=delta)
    text = chat(cfg["meta"]["lesson_model"], "", prompt, cfg, max_output_tokens=LESSON_MAX_TOKENS,
                effort=cfg["meta"]["lesson_effort"])
    line = " ".join(text.split())
    return (line or f"{strategy}: {rationale} -> {outcome}, dev change {delta}")[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="propose and print diffs; never trains")
    ap.add_argument("--strategy", default="tune_optimizer", choices=list(STRATEGIES))
    args = ap.parse_args()
    cfg = C.load()
    train_py = open(C.abspath(cfg["paths"]["candidate"])).read()
    ctx = {"train_py": train_py, "skills_md": open(C.abspath(cfg["paths"]["skills"])).read(),
           "recent": history.summaries(cfg["loop"]["recent_attempts_in_prompt"], cfg), "last_error": None}
    pairs = propose(args.strategy, ctx, cfg)
    print(f"valid candidates: {len(pairs)} of {len(ctx['candidates_log'])}")
    for cand, new in pairs:
        print(f"\n=== {cand.id}: {cand.rationale}\n{diff_text(train_py, new)}")


if __name__ == "__main__":
    main()
