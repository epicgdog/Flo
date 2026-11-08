"""JSON API + static host for the React dashboard. Run: uv run python3 dashboard/server.py [--port 8502]

Read-only, stdlib HTTP server. Same sources as dashboard/app.py: runs/state.json, runs/events.jsonl,
history/*/ files, history/baseline.json, skills/skills.md, runs/sim.*. Serves the built frontend from
dashboard/web/dist. All readers tolerate missing or half-written files, so it works while loop.py runs.
"""
import argparse
import glob
import json
import mimetypes
import os
import re
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import config as C  # noqa: E402

CFG = C.load()
P = CFG["paths"]
HIST = C.abspath(P["history"])
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")
FAILED = {"static_fail", "train_crash", "timeout", "load_fail", "too_large", "nan", "implausible",
          "no_valid_candidates"}
ITER_RE = re.compile(r"^\d{4}$")
LABELS = ("dev", "dev2", "heldout")


# ---------- readers ----------
def rjson(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def rjsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def rtext(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return None


def hfile(iter_id, name):
    return os.path.join(HIST, iter_id, name)


def status_of(o):
    if o.get("accepted"):
        return "accepted"
    r = str(o.get("reason") or "")
    if r == "killed_early":
        return "killed_early"
    if r in FAILED or r.startswith("error"):
        return "failed"
    return "rejected"


def outcomes():
    outs = []
    for f in sorted(glob.glob(os.path.join(HIST, "[0-9]" * 4, "outcome.json"))):
        o = rjson(f)
        if o:
            o["status"] = status_of(o)
            outs.append(o)
    return outs


def loss_log(iter_id, label="dev"):
    return [[r.get("elapsed_s"), r.get("loss")] for r in rjsonl(hfile(iter_id, f"train_log_{label}.jsonl"))]


def snap_cost(outs, evs):
    strat = {}
    for o in outs:
        s = strat.setdefault(o.get("strategy") or "-", {"attempts": 0, "accepted": 0})
        s["attempts"] += 1
        s["accepted"] += int(bool(o.get("accepted")))
    calls = [c for o in outs for c in (rjson(hfile(o["iter_id"], "snap.json"), []) or [])]
    fb = {}
    for c in calls:
        d = fb.setdefault(str(c.get("decision")), [0, 0])
        d[0] += 1
        d[1] += int(bool(c.get("used_fallback")))
    audits = [e for e in evs if e.get("type") == "snap_audit" and e.get("top_score") is not None]
    wins = [e for e in audits if e.get("bottom_score") is None or e["top_score"] < e["bottom_score"]]
    saved = 0.0
    for o in outs:
        for label in LABELS:
            rep = rjson(hfile(o["iter_id"], f"report_{label}.json"))
            if rep and rep.get("status") == "killed_early" and rep.get("train"):
                saved += CFG["train"]["time_budget_s"] - (rep["train"].get("elapsed_s") or 0)
    llm = [e for e in evs if e.get("type") == "llm_call"]
    return {
        "strategies": strat,
        "fallback_rate": {k: v[1] / v[0] for k, v in fb.items()},
        "audits": len(audits), "audit_wins": len(wins),
        "minutes_saved": saved / 60,
        "llm_calls": len(llm),
        "llm_tokens": sum((e.get("input_tokens") or 0) + (e.get("output_tokens") or 0) for e in llm),
        "llm_mean_s": (sum(e.get("seconds") or 0 for e in llm) / len(llm)) if llm else 0,
    }


def summary():
    s = rjson(C.abspath(P["state"]), {}) or {}
    b = rjson(os.path.join(HIST, "baseline.json"), {}) or {}
    outs = outcomes()
    evs = rjsonl(C.abspath(P["events"]))
    attempts = [o for o in outs if o.get("iter_id") != "0000"]
    live_iter = s.get("iter_id") if str(s.get("phase", "")).startswith("training") else None
    logs = {o["iter_id"]: loss_log(o["iter_id"]) for o in outs}
    if live_iter and live_iter not in logs:
        for label in ("heldout", "dev2", "dev"):
            lg = loss_log(live_iter, label)
            if lg:
                logs[live_iter] = lg
                break
    phase_label = {"training_dev": "dev", "training_dev_2": "dev2", "training_heldout": "heldout"}.get(s.get("phase"))
    live_log = loss_log(s["iter_id"], phase_label) if s.get("iter_id") and phase_label else []
    snap_checks = [e for e in evs if e.get("type") == "snap_check" and e.get("iter_id") == s.get("iter_id")
                   and e.get("run") == phase_label]
    iter_end = {e["iter_id"]: e["ts"] for e in evs if e.get("type") == "iter_end" and e.get("iter_id")}
    skills_path = C.abspath(P["skills"])
    sim_rows = rjsonl(C.abspath(P["sim_log"]))
    return {
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "state": s,
        "baseline": {"dev": b.get("mean_dev"), "heldout": b.get("mean_heldout")},
        "base_outcome": next((o for o in outs if o.get("iter_id") == "0000"), None),
        "attempts": attempts,
        "loss_logs": logs,
        "live": {"label": phase_label, "log": live_log, "snap_checks": snap_checks},
        "iter_end": iter_end,
        "events_tail": evs[-30:][::-1],
        "budget_s": CFG["train"]["time_budget_s"],
        "skills": rtext(skills_path),
        "snap_cost": snap_cost(attempts, evs),
        "sim": {"rows": sim_rows[-2000:], "has_frame": os.path.exists(C.abspath(P["sim_frame"]))},
        "models": tagged_models(),
        "bones": bones(),
    }


def dataset(set_name: str, n: int) -> dict:
    """First n samples of a dataset as the camera saw them: the 2D pose in pixels, the drone pose relative to the
    person, and the 3D pose in camera coordinates. Pixel coords are rebuilt from the stored normalized keypoints
    and the frame (root position and bbox size, in focal units)."""
    import numpy as np
    from checker.datagen import BONES
    if set_name not in _data_cache:
        d = np.load(C.abspath(P[set_name]))
        _data_cache[set_name] = {k: d[k] for k in ("kp2d", "mask", "frame", "pose3d", "cam")}
    d = _data_cache[set_name]
    W, H = CFG["data"]["sensor_px"]
    out = []
    for i in range(min(n, len(d["cam"]))):
        cam = d["cam"][i].astype(np.float64)
        az, el, dist, focal, yaw_off, pitch, roll = cam
        frame = d["frame"][i].astype(np.float64)
        mask = d["mask"][i]
        root2d = frame[:2] * focal + np.array([W / 2, H / 2])
        diag = frame[2] * focal
        uv = d["kp2d"][i].astype(np.float64) * diag + root2d
        pos = dist * np.array([np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(az)),
                               np.cos(np.deg2rad(el)) * np.sin(np.deg2rad(az)), np.sin(np.deg2rad(el))])
        out.append({
            "idx": i,
            "uv": [[float(x), float(y)] if m > 0 else None for (x, y), m in zip(uv, mask)],
            "mask": [int(m > 0) for m in mask],
            "visible": int((mask > 0).sum()),
            "pose3d": d["pose3d"][i].astype(float).tolist(),
            "cam": {"azimuth_deg": az, "elevation_deg": el, "distance_m": dist, "focal_px": focal,
                    "yaw_offset_deg": yaw_off, "pitch_down_deg": pitch, "roll_deg": roll,
                    "heading_deg": az + 180 + yaw_off, "pos": pos.tolist()},
        })
    return {"set": set_name, "W": W, "H": H, "bones": [list(map(int, b)) for b in BONES], "samples": out}


def bones():
    from checker.datagen import BONES
    return [list(map(int, b)) for b in BONES]


def attempt(iter_id):
    return {
        "iter_id": iter_id,
        "diff": rtext(hfile(iter_id, "chosen.diff")),
        "reports": {lb: rjson(hfile(iter_id, f"report_{lb}.json")) for lb in LABELS},
        "images": [lb for lb in ("dev", "heldout") if os.path.exists(hfile(iter_id, f"skeletons_{lb}.png"))],
        "logs": {lb: loss_log(iter_id, lb) for lb in LABELS},
        "snap": rjson(hfile(iter_id, "snap.json")),
        "lesson": rtext(hfile(iter_id, "lesson.txt")),
    }


# ---------- 3D viewer (lazy torch import; only runs on request) ----------
def tagged_models():
    return {o["tag"]: o["iter_id"] for o in outcomes()
            if o.get("accepted") and o.get("tag") and os.path.exists(hfile(o["iter_id"], "model/model.pt"))}


_model_cache, _model_lock, _data_cache = {}, threading.Lock(), {}


def viewer(which, idx, elev):
    import numpy as np
    from checker.check import load_model, predict
    from checker.datagen import BONES, camera_extrinsics, project
    from checker.metrics import mpjpe_mm

    models = tagged_models()
    if "v0" not in models:
        return {"error": "No v0 model yet."}
    latest = max(models, key=lambda t: int(t[1:]))
    if which not in _data_cache:
        d = np.load(C.abspath(P[which]))
        _data_cache[which] = {"pose3d": d["pose3d"], "cam": d["cam"]}
    d = _data_cache[which]
    n = int(len(d["pose3d"]))
    idx = max(0, min(n - 1, idx))
    cam = d["cam"][idx].astype(np.float64)
    R, _ = camera_extrinsics(cam[None])
    world = d["pose3d"][idx].astype(np.float64) @ R[0]
    new_cam = np.array([[cam[0], elev, cam[2], cam[3], 0.0, elev, 0.0]], dtype=np.float32)
    view = project(world[None], new_cam, np.random.default_rng(0), CFG)
    gt = view["pose3d"][0].astype(np.float64)
    out = {"n": n, "idx": idx, "bones": [list(map(int, b)) for b in BONES], "gt": gt.tolist(), "preds": []}
    with _model_lock:
        for tag in dict.fromkeys(("v0", latest)):
            if tag not in _model_cache:
                _model_cache[tag] = load_model(hfile(models[tag], "model"))[0]
            hyps = predict(_model_cache[tag], view["kp2d"], view["mask"], view["frame"], CFG)[0]
            pred = hyps.mean(axis=0)
            out["preds"].append({
                "tag": tag, "latest": tag == latest, "pred": pred.tolist(),
                "mpjpe": float(mpjpe_mm(pred[None], gt[None])),
                "hyps": hyps[:10].tolist() if tag == latest else [],
            })
    return out


# ---------- HTTP ----------
def clean(x):
    """NaN/inf -> None so the browser's JSON.parse accepts the payload."""
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(clean(obj), default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, cache="no-store"):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path == "/api/summary":
                return self.send_json(summary())
            if len(parts) == 3 and parts[:2] == ["api", "attempt"] and ITER_RE.match(parts[2]):
                return self.send_json(attempt(parts[2]))
            if len(parts) == 4 and parts[:2] == ["api", "image"] and ITER_RE.match(parts[2]) \
                    and parts[3] in ("skeletons_dev.png", "skeletons_heldout.png"):
                return self.send_file(hfile(parts[2], parts[3]))
            if u.path == "/api/sim_frame":
                return self.send_file(C.abspath(P["sim_frame"]))
            if u.path == "/api/dataset":
                which = q.get("set", "train")
                if which not in ("train", "dev", "heldout"):
                    return self.send_json({"error": "bad set"}, 400)
                return self.send_json(dataset(which, max(1, min(int(q.get("n", 2)), 20))))
            if u.path == "/api/viewer":
                which = q.get("set", "dev")
                if which not in ("dev", "heldout"):
                    return self.send_json({"error": "bad set"}, 400)
                return self.send_json(viewer(which, int(q.get("idx", 0)), float(q.get("elev", 10))))
        except Exception as e:  # keep the server up whatever the files look like
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        if parts and parts[0] == "api":
            return self.send_error(404)
        # static frontend (SPA fallback to index.html)
        rel = os.path.normpath(u.path.lstrip("/")) if u.path != "/" else "index.html"
        path = os.path.join(DIST, rel)
        if not os.path.abspath(path).startswith(DIST + os.sep) or not os.path.isfile(path):
            path = os.path.join(DIST, "index.html")
        if not os.path.exists(path):
            return self.send_error(503, "frontend not built: cd dashboard/web && npm run build")
        self.send_file(path, "public, max-age=31536000, immutable" if "/assets/" in u.path else "no-store")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"dashboard on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
