"""Snap decider: route, rank, should_stop, triage. All requests go through _call().

Any exception, timeout (snap.timeout_s) or answer outside the options uses the fallback.
Snap is disabled (snap.enabled: false; no SUBQ_API_KEY and no known request format), so every decision
falls back without calling _call. Milestone M9 only needs to fill in _call.
"""
import os
import subprocess
import time

from harness import config as C
from harness import events, history

TRIAGE_FROM_STATUS = {
    "train_crash": "crash", "static_fail": "crash", "load_fail": "crash", "too_large": "crash",
    "timeout": "timeout", "nan": "diverged", "implausible": "implausible_pose",
    "killed_early": "no_gain", "no_gain": "no_gain", "heldout_regression": "heldout_regression",
}

CALLS: list[dict] = []   # {decision, options, choice, used_fallback, latency_s}; the loop writes them to snap.json


def drain_calls() -> list[dict]:
    out = list(CALLS)
    CALLS.clear()
    return out


def build_context(cfg: dict) -> str:
    """Coder-safe history summaries + skills.md + best train.py (the one on main). Never held-out metrics."""
    lines = ["ATTEMPTS:"]
    for s in history.summaries(10 ** 6, cfg):
        lines.append(f"{s['id']} | {s['strategy']} | {s['rationale']} | dev={s['dev']} | {s['result']}")
    skills = open(C.abspath(cfg["paths"]["skills"])).read()
    best_py = subprocess.run(["git", "show", f"main:{cfg['paths']['candidate']}"], cwd=C.ROOT,
                             capture_output=True, text=True).stdout
    return "\n".join(lines + ["", "SKILLS:", skills, "", "BEST train.py:", best_py])


def _call(question: str, options: list[str], context: str) -> str:
    """The only code that knows Snap's request format (SUBQ_API_KEY, snap.base_url, snap.model, snap.timeout_s)."""
    cfg = C.load()
    if not os.environ.get("SUBQ_API_KEY") or cfg["snap"]["base_url"] == "SET_ME":
        raise RuntimeError("Snap not configured")
    raise NotImplementedError("Snap request format not known yet (milestone M9)")


def _decide(decision: str, question: str, options: list[str], context: str, cfg: dict, fallback: str):
    t0 = time.time()
    used = True
    choice = fallback
    if cfg["snap"]["enabled"]:
        try:
            answer = _call(question, options, context)
            if answer in options:
                choice, used = answer, False
        except Exception:
            pass
    CALLS.append({"decision": decision, "options": options, "choice": choice,
                  "used_fallback": used, "latency_s": round(time.time() - t0, 3)})
    return choice, used


def route(context: str, options: list[str], cfg: dict) -> tuple[str, bool]:
    """Fallback: round-robin by iteration number (the current iter_id in runs/state.json)."""
    n = int(events.read_state(cfg).get("iter_id") or 1)
    return _decide("route", "Which strategy is most likely to lower dev error next?", options, context, cfg,
                   options[(n - 1) % len(options)])


def rank(context: str, candidates: list[str], cfg: dict) -> tuple[list[int], bool]:
    """Ask repeatedly, removing the winner each time, to build a full order. Fallback: original order."""
    remaining = list(range(len(candidates)))
    order, any_fallback = [], False
    while remaining:
        opts = [str(i) for i in remaining]
        ctx = context + "\n\nCANDIDATES:\n" + "\n\n".join(f"[{i}] {candidates[i]}" for i in remaining)
        choice, used = _decide("rank", "Which candidate is most likely to lower dev error?", opts, ctx, cfg, opts[0])
        any_fallback = any_fallback or used
        order.append(int(choice))
        remaining.remove(int(choice))
    return order, any_fallback


def should_stop(context: str, log_tail: str, cfg: dict) -> tuple[bool, bool]:
    choice, used = _decide("should_stop", "Should this training run be stopped now?", ["continue", "stop"],
                           context + "\n\nCURRENT train_log.jsonl:\n" + log_tail, cfg, "continue")
    return choice == "stop", used


def triage(context: str, report: dict, options: list[str], cfg: dict) -> tuple[str, bool]:
    """Fallback: map from report status."""
    fb = TRIAGE_FROM_STATUS.get(report.get("status"), "crash")
    return _decide("triage", "What best describes this failure?", options,
                   context + f"\n\nREPORT: status={report.get('status')} error={report.get('error')}", cfg, fb)
