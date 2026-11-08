"""Autonomous loop. One iteration: route, propose, rank, commit, dev seed 0, dev seed 1, held-out, gate,
triage, lesson. Only gate.accept() (on checker numbers) moves main.

CLI: uv run python3 loop.py --hours 3 [--no-snap] [--max-iters N]
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta

from agents import coder, llm, snap
from checker.check import check, evaluate
from harness import config as C
from harness import events, gate, history

MIN_LEFT_S = 8 * 60          # section 12: no new iteration with less than 8 minutes left
PROGRESS_EVERY_S = 10        # section 12: train_progress every 10 s
TRIAGE_OPTIONS = ["crash", "timeout", "diverged", "implausible_pose", "no_gain", "heldout_regression"]
WORK = os.path.join(C.ROOT, "runs", "work")


def _git(*args, check_ok=True):
    return gate._git(*args, check=check_ok)


def _hist(cfg, iter_id, name=""):
    return os.path.join(C.abspath(cfg["paths"]["history"]), iter_id, name)


def counts(cfg) -> dict:
    outs = [o for o in history.outcomes(cfg) if o.get("iter_id") != "0000"]
    return {"iters": len(outs),
            "accepted": sum(1 for o in outs if o.get("accepted")),
            "rejected": sum(1 for o in outs if not o.get("accepted")),
            "killed_early": sum(1 for o in outs if o.get("reason") == "killed_early"),
            "errors": sum(1 for o in outs if str(o.get("reason") or "").startswith("error"))}


def baseline() -> dict:
    b = {}
    p = os.path.join(C.abspath("history"), "baseline.json")
    if os.path.exists(p):
        with open(p) as f:
            b = json.load(f)
    return {"dev": b.get("mean_dev"), "heldout": b.get("mean_heldout")}


def last_error(cfg) -> str | None:
    """The `error` field of the most recent report that has one."""
    for d in reversed(history.iter_ids(cfg)):
        for name in ("report_heldout.json", "report_dev2.json", "report_dev.json"):
            rep = history.load(d, name, cfg)
            if rep:
                if rep.get("error"):
                    return rep["error"]
                break
    return None


def append_lesson(cfg, cls: str, line: str) -> None:
    """Put the lesson in the section for its class. Crashes and inconclusive results are not bans."""
    prefix = {"pass": "## Confirmed", "regression": "## Failed ideas", "heldout_regression": "## Failed ideas",
              "inconclusive": "## Inconclusive", "crash": "## Untested"}[cls]
    path = C.abspath(cfg["paths"]["skills"])
    lines = open(path).read().rstrip("\n").split("\n")
    head = next((i for i, x in enumerate(lines) if x.startswith(prefix)), None)
    if head is None:
        lines += ["", prefix + (" (not evaluated; not a ban)" if cls == "crash" else
                                " (within seed noise; not a ban)" if cls == "inconclusive" else "")]
        head = len(lines) - 1
    end = next((i for i in range(head + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    lines.insert(end, line)
    open(path, "w").write("\n".join(lines) + "\n")


def recent_loss_curves(cfg, n=5) -> str:
    """Final loss curves of the last n completed dev runs, for should_stop."""
    out = []
    for d in reversed(history.iter_ids(cfg)):
        log = history.load(d, "train_log_dev.jsonl", cfg)
        if log:
            pts = [json.loads(x) for x in log.splitlines() if x.strip()]
            out.append(f"{d}: " + ", ".join(f"{p['elapsed_s']:.0f}s={p['loss']:.4f}" for p in pts if p.get("loss") is not None))
        if len(out) >= n:
            break
    return "\n".join(out)


class Monitor(threading.Thread):
    """During a training run: copy the log into history, emit train_progress, ask snap.should_stop."""

    def __init__(self, cfg, iter_id, label, work_dir, context, stop_event):
        super().__init__(daemon=True)
        self.cfg, self.iter_id, self.label, self.work_dir = cfg, iter_id, label, work_dir
        self.context, self.stop_event = context, stop_event
        self.done = threading.Event()
        self.t0 = time.time()

    def run(self):
        lc = self.cfg["loop"]
        src = os.path.join(self.work_dir, "train_log.jsonl")
        dst = _hist(self.cfg, self.iter_id, f"train_log_{self.label}.jsonl")
        last_progress, last_check, seen = 0.0, None, 0
        while not self.done.wait(1.0):
            if os.path.exists(src):
                shutil.copy(src, dst)
                rows = [json.loads(x) for x in open(src).read().splitlines() if x.strip()]
                if rows and len(rows) != seen and time.time() - last_progress >= PROGRESS_EVERY_S - 1:
                    seen, last_progress = len(rows), time.time()
                    r = rows[-1]
                    events.emit("train_progress", self.cfg, iter_id=self.iter_id, run=self.label, **r)
                    events.set_state(self.cfg, train_elapsed_s=r.get("elapsed_s"), train_loss=r.get("loss"))
            el = time.time() - self.t0
            if el >= lc["early_stop_min_s"] and (last_check is None or time.time() - last_check >= lc["early_stop_check_s"]):
                last_check = time.time()
                log_tail = open(src).read() if os.path.exists(src) else ""
                ctx = self.context + "\n\nLAST 5 COMPLETED RUNS (final loss curves):\n" + recent_loss_curves(self.cfg)
                stop, used = snap.should_stop(ctx, log_tail, self.cfg)
                events.emit("snap_check", self.cfg, iter_id=self.iter_id, run=self.label,
                            elapsed_s=round(el, 1), answer="stop" if stop else "continue", used_fallback=used)
                if stop:
                    self.stop_event.set()
                    return


def train_stage(cfg, iter_id, set_name, seed, label, phase, context) -> dict:
    events.set_state(cfg, phase=phase, train_elapsed_s=0.0, train_loss=None)
    work = os.path.join(WORK, f"{iter_id}_{label}")
    shutil.rmtree(work, ignore_errors=True)
    stop_event = threading.Event()
    mon = Monitor(cfg, iter_id, label, work, context, stop_event)
    events.emit("train_started", cfg, iter_id=iter_id, run=label, set=set_name, seed=seed)
    mon.start()
    try:
        rep = check(C.abspath(os.path.dirname(cfg["paths"]["candidate"])), set_name, seed, iter_id, work, cfg, stop_event)
    finally:
        mon.done.set()
        mon.join(timeout=10)
    if os.path.exists(os.path.join(work, "train_log.jsonl")):
        shutil.copy(os.path.join(work, "train_log.jsonl"), _hist(cfg, iter_id, f"train_log_{label}.jsonl"))
    if label in ("dev", "heldout") and os.path.exists(os.path.join(work, "skeletons.png")):
        shutil.copy(os.path.join(work, "skeletons.png"), _hist(cfg, iter_id, f"skeletons_{label}.png"))
        rep["skeletons_png"] = os.path.relpath(_hist(cfg, iter_id, f"skeletons_{label}.png"), C.ROOT)
    history.save(iter_id, f"report_{label}.json", rep, cfg)
    if rep["status"] == "killed_early":
        events.emit("killed_early", cfg, iter_id=iter_id, run=label)
    else:
        events.emit("train_finished", cfg, iter_id=iter_id, run=label, status=rep["status"], score=rep["score"])
    return rep


CRASH_STATUSES = {"static_fail", "train_crash", "load_fail", "too_large", "timeout", "nan"}


def run_trials(cfg, iter_id, pairs) -> tuple[list[dict], dict | None]:
    """Short dev trial (seed 0, loop.trial_budget_s) for every valid candidate, one at a time.
    Returns (rows, best ok row or None). Trial scores are only comparable with each other, not with the incumbent."""
    budget = cfg["loop"]["trial_budget_s"]
    tcfg = {**cfg, "train": {**cfg["train"], "time_budget_s": budget, "kill_after_s": budget + 15}}
    rows = []
    for cand, new_py in pairs:
        d = os.path.join(WORK, f"{iter_id}_trial{cand.id}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        with open(os.path.join(d, "train.py"), "w") as f:
            f.write(new_py)
        events.set_state(cfg, phase=f"trial_{cand.id}", train_elapsed_s=0.0, train_loss=None)
        rep = check(d, "dev", 0, iter_id, os.path.join(d, "out"), tcfg)
        rows.append({"id": cand.id, "status": rep["status"], "score": rep["score"], "error": rep.get("error")})
        events.emit("trial_finished", cfg, iter_id=iter_id, candidate=cand.id, status=rep["status"], score=rep["score"])
    history.save(iter_id, "trials.json", rows, cfg)
    ok = [r for r in rows if r["status"] == "ok"]
    return rows, (min(ok, key=lambda r: r["score"]) if ok else None)


def incumbent_score(cfg, best: dict, seed: int) -> float:
    """Incumbent's dev score on a seed it has not been run on yet. Trained once, then cached in history."""
    owner = best["iter_id"] or "0000"
    cached = history.load(owner, f"incumbent_seed{seed}.json", cfg)
    if cached:
        return cached["score"]
    code = subprocess.run(["git", "show", f"{best['tag']}:{cfg['paths']['candidate']}"], cwd=C.ROOT,
                          capture_output=True, text=True, check=True).stdout
    d = os.path.join(WORK, f"incumbent_{best['tag']}_seed{seed}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "train.py"), "w") as f:
        f.write(code)
    rep = check(d, "dev", seed, owner, os.path.join(d, "out"), cfg)
    shutil.rmtree(d, ignore_errors=True)
    if rep["status"] != "ok":
        raise RuntimeError(f"incumbent seed {seed} failed: {rep['status']} {rep.get('error')}")
    history.save(owner, f"incumbent_seed{seed}.json", {"tag": best["tag"], "score": rep["score"]}, cfg)
    return rep["score"]


def full_eval(cfg, iter_id, context, best, out) -> tuple[str, str | None, str]:
    """Full pipeline for the committed candidate. Returns (class, reason, detail) and fills out[dev/dev2/dev3/heldout].
    One training run per seed. Held-out is scored on the seed-0 checkpoint that dev scored, so there is no second run."""
    rep = train_stage(cfg, iter_id, "dev", 0, "dev", "training_dev", context)
    if rep["status"] in CRASH_STATUSES:
        return "crash", rep["status"], rep.get("error") or rep["status"]
    if rep["status"] != "ok":
        return "regression", rep["status"], rep.get("error") or rep["status"]
    out["dev"] = rep["score"]
    work0 = os.path.join(WORK, f"{iter_id}_dev")
    hev = evaluate(work0, C.abspath(cfg["paths"]["heldout"]), cfg)
    history.save(iter_id, "report_heldout.json", {"iter_id": iter_id, "set": "heldout", "seed": 0,
                                                  "from_checkpoint": "dev_seed0", "status": hev["status"],
                                                  "score": hev["score"], "metrics": hev["metrics"],
                                                  "error": hev["error"], "params": hev["params"]}, cfg)
    if hev["status"] in CRASH_STATUSES:
        return "crash", hev["status"], hev["error"] or hev["status"]
    if hev["status"] != "ok":
        return "regression", hev["status"], hev["error"] or hev["status"]
    out["heldout"] = hev["score"]

    d0 = out["dev"] - best["dev"]
    if gate.classify_dev([d0], cfg) == "regression":
        return "regression", "no_gain", f"seed0 paired dev diff {d0:+.2f} mm"

    rep2 = train_stage(cfg, iter_id, "dev", 1, "dev2", "training_dev_2", context)
    if rep2["status"] in CRASH_STATUSES or rep2["status"] != "ok":
        return "crash", rep2["status"], rep2.get("error") or rep2["status"]
    out["dev2"] = rep2["score"]
    inc2 = best["dev2"] if best.get("dev2") is not None else incumbent_score(cfg, best, 1)
    diffs = [d0, out["dev2"] - inc2]
    cls = gate.classify_dev(diffs, cfg)
    if cls == "near_miss":                      # leaning better: confirm on a third seed, incumbent included
        rep3 = train_stage(cfg, iter_id, "dev", 2, "dev3", "training_dev_3", context)
        if rep3["status"] in CRASH_STATUSES or rep3["status"] != "ok":
            return "crash", rep3["status"], rep3.get("error") or rep3["status"]
        out["dev3"] = rep3["score"]
        diffs.append(out["dev3"] - incumbent_score(cfg, best, 2))
        cls = gate.classify_dev(diffs, cfg)
    detail = f"paired dev diffs mm {[round(x, 2) for x in diffs]}"
    if cls == "regression":
        return "regression", "no_gain", detail
    if cls != "pass":
        return "inconclusive", "no_gain", detail
    if not gate.passes_heldout(out["heldout"], best["heldout"], cfg):
        return "heldout_regression", "heldout_regression", detail
    return "pass", None, detail


def promote(iter_id: str, cfg) -> None:
    """The artifact that was scored on both splits: the dev seed-0 checkpoint."""
    mdir = _hist(cfg, iter_id, "model")
    os.makedirs(mdir, exist_ok=True)
    for fn in ("model.pt", "model_def.py"):
        shutil.copy(os.path.join(WORK, f"{iter_id}_dev", fn), os.path.join(mdir, fn))


def run_iteration(iter_id: str, cfg: dict, use_snap: bool) -> dict:
    """Returns outcome.json content."""
    cfg = {**cfg, "snap": {**cfg["snap"], "enabled": bool(use_snap and cfg["snap"]["enabled"])}}
    t0 = time.time()
    tokens0 = llm.TOKENS["input"] + llm.TOKENS["output"]
    best = history.best(cfg)
    snap.drain_calls()
    out = {"iter_id": iter_id, "strategy": None, "rationale": None, "accepted": False, "reason": None,
           "cls": None, "tag": None, "dev": None, "dev2": None, "dev3": None, "heldout": None,
           "best_dev_before": best["dev"], "best_heldout_before": best["heldout"], "trials": None,
           "triage": None, "lesson": None, "duration_s": None, "llm_tokens": None}
    calls: list[dict] = []

    def save_snap():
        calls.extend(snap.drain_calls())
        history.save(iter_id, "snap.json", calls, cfg)

    # 1
    gate.start_iteration(iter_id)
    events.emit("iter_start", cfg, iter_id=iter_id)
    events.set_state(cfg, iter_id=iter_id, phase="starting", strategy=None, rationale=None,
                     train_elapsed_s=None, train_loss=None, best=best)
    lesson_due, cls, detail = False, None, ""
    try:
        train_py = open(C.abspath(cfg["paths"]["candidate"])).read()
        context = snap.build_context(cfg)
        # 2
        events.set_state(cfg, phase="routing")
        strategy, _ = snap.route(context, cfg["loop"]["strategies"], cfg)
        out["strategy"] = strategy
        history.save(iter_id, "strategy.json", {"strategy": strategy}, cfg)
        events.emit("strategy_chosen", cfg, iter_id=iter_id, strategy=strategy)
        events.set_state(cfg, strategy=strategy)
        save_snap()
        # 3
        events.set_state(cfg, phase="proposing")
        ctx = {"train_py": train_py, "skills_md": open(C.abspath(cfg["paths"]["skills"])).read(),
               "recent": history.summaries(cfg["loop"]["recent_attempts_in_prompt"], cfg),
               "last_error": last_error(cfg)}
        pairs = coder.propose(strategy, ctx, cfg)
        history.save(iter_id, "candidates.json",
                     {"candidates": ctx["candidates_log"], "raw_responses": ctx["raw_responses"]}, cfg)
        events.emit("candidates_proposed", cfg, iter_id=iter_id, n_valid=len(pairs))
        if not pairs:
            out["reason"] = "no_valid_candidates"
            gate.reject(iter_id)
            events.emit("rejected", cfg, iter_id=iter_id, reason="no_valid_candidates")
            return _finish(cfg, out, t0, tokens0, calls)
        # 4 ranking (Snap); only used to pick the candidate when trials are off
        events.set_state(cfg, phase="ranking")
        texts = [f"id={c.id}\nrationale: {c.rationale}\n{coder.diff_text(train_py, new)}" for c, new in pairs]
        order, _ = snap.rank(context, texts, cfg)
        history.save(iter_id, "ranking.json", {"order": [pairs[i][0].id for i in order]}, cfg)
        events.emit("candidates_ranked", cfg, iter_id=iter_id, order=[pairs[i][0].id for i in order])
        save_snap()
        chosen_idx = order[0]
        # 5 short trials: every valid candidate gets the same short budget; only the best one gets a full run
        if cfg["loop"]["trials"]:
            trial_rows, winner = run_trials(cfg, iter_id, pairs)
            out["trials"] = trial_rows
            if winner is None:
                errs = [r["error"] for r in trial_rows if r.get("error")]
                detail = str(errs[0]) if errs else "all trials failed"
                history.save(iter_id, "report_dev.json", {"status": "trials_failed", "error": detail}, cfg)
                cls, out["reason"] = "crash", "trials_failed"
            else:
                chosen_idx = next(i for i, (c, _) in enumerate(pairs) if c.id == winner["id"])
        chosen, chosen_py = pairs[chosen_idx]
        out["rationale"] = chosen.rationale
        lesson_due = True
        if cls is None:
            # 6 commit and full evaluation
            events.set_state(cfg, phase="committing", rationale=chosen.rationale)
            with open(C.abspath(cfg["paths"]["candidate"]), "w") as f:
                f.write(chosen_py)
            gate.commit_candidate(iter_id, chosen.rationale)
            history.save(iter_id, "chosen.diff", _git("diff", "main", "--", cfg["paths"]["candidate"]) + "\n", cfg)
            events.emit("candidate_committed", cfg, iter_id=iter_id, candidate=chosen.id)
            cls, out["reason"], detail = full_eval(cfg, iter_id, context, best, out)
            save_snap()
        # 9
        events.set_state(cfg, phase="deciding")
        out["cls"] = cls
        if cls == "pass":
            tag = gate.accept(iter_id, out["dev"], out["heldout"])
            out.update(accepted=True, tag=tag, reason=None)
            promote(iter_id, cfg)
            events.emit("accepted", cfg, iter_id=iter_id, tag=tag, dev=out["dev"], heldout=out["heldout"])
        else:
            gate.reject(iter_id)
            events.emit("rejected", cfg, iter_id=iter_id, reason=out["reason"], cls=cls)
            # 10
            events.set_state(cfg, phase="triaging")
            out["triage"], _ = snap.triage(context, {"status": out["reason"], "error": detail}, TRIAGE_OPTIONS, cfg)
            events.emit("triaged", cfg, iter_id=iter_id, triage=out["triage"])
            save_snap()
        # audit (Snap on): train and dev-score the lowest-ranked candidate; not gated, not merged
        if cfg["snap"]["enabled"] and len(order) > 1 and int(iter_id) % cfg["loop"]["snap_audit_every"] == 0:
            low, low_py = pairs[order[-1]]
            adir = os.path.join(WORK, f"{iter_id}_audit")
            os.makedirs(adir, exist_ok=True)
            open(os.path.join(adir, "train.py"), "w").write(low_py)
            arep = check(adir, "dev", 0, iter_id, os.path.join(adir, "out"), cfg)
            events.emit("snap_audit", cfg, iter_id=iter_id, top_id=chosen.id, top_score=out["dev"],
                        bottom_id=low.id, bottom_score=arep["score"])
    except Exception as e:
        out.update(accepted=False, reason=f"error: {e!r}")
        lesson_due = False
        events.emit("error", cfg, iter_id=iter_id, error=repr(e))
        _git("checkout", "--", cfg["paths"]["candidate"], check_ok=False)
        if _git("branch", "--list", f"iter/{iter_id}", check_ok=False):
            gate.reject(iter_id)
    # 11 lesson: class decides the section, and the detail (numbers or the real error) goes to the writer
    if lesson_due:
        try:
            events.set_state(cfg, phase="learning")
            delta = None if out["dev"] is None or best["dev"] is None else out["dev"] - best["dev"]
            lesson = coder.write_lesson(iter_id, out["strategy"], out["rationale"], cls, detail, delta, cfg)
            out["lesson"] = lesson
            history.save(iter_id, "lesson.txt", lesson + "\n", cfg)
            append_lesson(cfg, cls, f"- [{iter_id}] {lesson}")
            _git("add", cfg["paths"]["skills"])
            _git("commit", "-m", f"lesson {iter_id}")
            events.emit("lesson_written", cfg, iter_id=iter_id, lesson=lesson)
        except Exception as e:
            events.emit("error", cfg, iter_id=iter_id, error=f"lesson: {e!r}")
    return _finish(cfg, out, t0, tokens0, calls)


def _finish(cfg, out, t0, tokens0, calls) -> dict:
    # 12
    iter_id = out["iter_id"]
    calls.extend(snap.drain_calls())
    history.save(iter_id, "snap.json", calls, cfg)
    out["duration_s"] = round(time.time() - t0, 1)
    out["llm_tokens"] = llm.TOKENS["input"] + llm.TOKENS["output"] - tokens0
    history.save(iter_id, "outcome.json", out, cfg)
    for d in os.listdir(WORK) if os.path.isdir(WORK) else []:
        if d.startswith(iter_id):
            shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
    events.emit("iter_end", cfg, iter_id=iter_id, duration_s=out["duration_s"])
    events.set_state(cfg, phase="idle", best=history.best(cfg), counts=counts(cfg))
    return out


def _kill_orphan_training(cfg) -> None:
    """A hard-killed loop leaves its checker train.py child running; it would steal CPU from the next run."""
    pat = "check_train_.*/train.py --data " + C.abspath(cfg["paths"]["train"])
    pids = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"killed orphan training process {pid}", flush=True)
        except (ProcessLookupError, ValueError):
            pass


def resume(cfg) -> None:
    """Kill orphan training, git checkout main, delete leftover iter/* branches."""
    _kill_orphan_training(cfg)
    _git("checkout", "--", cfg["paths"]["candidate"], cfg["paths"]["skills"], check_ok=False)
    _git("checkout", "main")
    for b in _git("branch", "--list", "iter/*").split():
        if b != "*":
            _git("branch", "-D", b)


def main() -> None:
    cfg = C.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=cfg["loop"]["hours"])
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args()
    if history.load("0000", "outcome.json", cfg) is None or cfg["gate"]["sigma_dev"] is None:
        raise SystemExit("no v0 baseline: run checker/calibrate.py (M4) first")
    resume(cfg)
    os.makedirs(WORK, exist_ok=True)
    start = datetime.now()
    ends = start + timedelta(hours=args.hours)
    events.set_state(cfg, run_id=start.strftime("%Y-%m-%dT%H-%M"), started_at=start.isoformat(timespec="seconds"),
                     ends_at=ends.isoformat(timespec="seconds"), iter_id=None, phase="idle", strategy=None,
                     rationale=None, train_elapsed_s=None, train_loss=None, best=history.best(cfg),
                     baseline=baseline(), counts=counts(cfg))
    n = 0
    while (ends - datetime.now()).total_seconds() >= MIN_LEFT_S:
        if args.max_iters is not None and n >= args.max_iters:
            break
        out = run_iteration(history.new_iter_id(cfg), cfg, use_snap=not args.no_snap)
        n += 1
        print(f"{out['iter_id']} {out['strategy']}: {'accepted ' + str(out['tag']) if out['accepted'] else 'rejected: ' + str(out['reason'])}",
              flush=True)
    # ends_at = actual stop time, so the dashboard shows "finished" (not "stalled") after an early stop
    events.set_state(cfg, phase="idle", ends_at=datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
