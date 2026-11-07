"""Read-only progress dashboard. Run: uv run streamlit run dashboard/app.py

Sources: runs/state.json, runs/events.jsonl, history/*/ files, history/baseline.json, skills/skills.md, runs/sim.*.
Auto-refreshes every 5 s with st.fragment(run_every=5). Works while loop.py runs and after it stops.
"""
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.datagen import BONES, project  # noqa: E402
from checker.metrics import mpjpe_mm  # noqa: E402
from harness import config as C  # noqa: E402

CFG = C.load()
P = CFG["paths"]
HIST = C.abspath(P["history"])
GREEN, RED, GREY, BLACK, BLUE, ORANGE = "#2e7d32", "#c62828", "#9e9e9e", "#111111", "#1565c0", "#ef6c00"
FAILED = {"static_fail", "train_crash", "timeout", "load_fail", "too_large", "nan", "implausible",
          "no_valid_candidates"}


# ---------- readers (all tolerant of missing or half-written files) ----------
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


def parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def state():
    return rjson(C.abspath(P["state"]), {}) or {}


def events():
    return rjsonl(C.abspath(P["events"]))


def outcomes():
    return [o for o in (rjson(f) for f in sorted(glob.glob(os.path.join(HIST, "[0-9]" * 4, "outcome.json")))) if o]


def attempts():
    return [o for o in outcomes() if o.get("iter_id") != "0000"]


def baseline():
    b = rjson(os.path.join(HIST, "baseline.json"), {}) or {}
    return b.get("mean_dev"), b.get("mean_heldout")


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


def pct(a, b):
    return f"{a - b:+.1f} mm ({100 * (a - b) / b:+.1f}%)" if a is not None and b else "n/a"


st.set_page_config(page_title="Drone Pose RSI", layout="wide")


# ---------- header ----------
@st.fragment(run_every=5)
def header():
    s = state()
    now = datetime.now()
    last, ends, started = parse_ts(s.get("last_update", "")), parse_ts(s.get("ends_at", "")), parse_ts(s.get("started_at", ""))
    if ends and now > ends:
        status, color = "finished", GREY
    elif last and (now - last).total_seconds() < 60:
        status, color = "running", GREEN
    elif s:
        status, color = "stalled", RED
    else:
        status, color = "no run yet", GREY
    bdev, bho = baseline()
    best = s.get("best") or {}
    cnt = s.get("counts") or {}
    st.markdown(f"### Drone Pose RSI harness &nbsp; <span style='color:{color}'>● {status}</span>", unsafe_allow_html=True)
    c = st.columns(6)
    el = (now - started).total_seconds() / 60 if started else 0
    rem = max(0.0, (ends - now).total_seconds() / 60) if ends else 0
    c[0].metric("Time", f"{el:.0f} min elapsed", f"{rem:.0f} min remaining", delta_color="off")
    c[1].metric("Now", f"{s.get('iter_id') or '-'} · {s.get('phase') or '-'}", s.get("strategy") or "", delta_color="off")
    c[2].metric("Best dev", f"{best['dev']:.1f} mm" if best.get("dev") is not None else "-",
                pct(best.get("dev"), bdev), delta_color="inverse")
    c[3].metric("Best held-out", f"{best['heldout']:.1f} mm" if best.get("heldout") is not None else "-",
                pct(best.get("heldout"), bho), delta_color="inverse")
    c[4].metric("Iterations", cnt.get("iters", 0), f"{cnt.get('accepted', 0)} accepted / {cnt.get('rejected', 0)} rejected",
                delta_color="off")
    c[5].metric("Killed early / errors", f"{cnt.get('killed_early', 0)} / {cnt.get('errors', 0)}")


# ---------- progress ----------
def loss_log(iter_id, label="dev"):
    return rjsonl(hfile(iter_id, f"train_log_{label}.jsonl"))


@st.fragment(run_every=5)
def progress_tab():
    outs = attempts()
    bdev, bho = baseline()
    if not outs:
        st.info("No attempts yet.")
        return
    devs = [o["dev"] for o in outs if o.get("dev") is not None] + [v for v in (bdev,) if v]
    top = max(devs) * 1.05 if devs else 1.0
    fig = go.Figure()
    for kind, color, symbol in (("accepted", GREEN, "circle"), ("rejected", RED, "circle"),
                                ("killed_early", GREY, "circle"), ("failed", BLACK, "x")):
        pts = [o for o in outs if status_of(o) == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[int(o["iter_id"]) for o in pts], y=[o["dev"] if o.get("dev") is not None else top for o in pts],
                mode="markers", name=kind, marker=dict(color=color, symbol=symbol, size=10),
                text=[f"{o['iter_id']} {o.get('strategy')}: {o.get('reason') or o.get('tag')}" for o in pts]))
    best_dev, best_ho, xs, yd, yh = None, None, [], [], []
    base = next((o for o in outcomes() if o.get("iter_id") == "0000"), {})
    best_dev, best_ho = base.get("dev"), base.get("heldout")
    for o in outs:
        if o.get("accepted"):
            best_dev, best_ho = o["dev"], o["heldout"]
        xs.append(int(o["iter_id"]))
        yd.append(best_dev)
        yh.append(best_ho)
    fig.add_trace(go.Scatter(x=xs, y=yd, mode="lines", line=dict(shape="hv", color=GREEN), name="best-so-far dev"))
    fig.add_trace(go.Scatter(x=xs, y=yh, mode="lines", line=dict(shape="hv", color=BLUE), name="best-so-far held-out"))
    if bdev:
        fig.add_hline(y=bdev, line_dash="dash", line_color=GREEN, annotation_text="baseline dev")
    if bho:
        fig.add_hline(y=bho, line_dash="dash", line_color=BLUE, annotation_text="baseline held-out")
    fig.update_layout(xaxis_title="iteration", yaxis_title="score (mm)", height=380, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Accepted changes")
    rows = [{"tag": o["tag"], "iteration": o["iter_id"], "rationale": o.get("rationale"),
             "dev change (mm)": None if o.get("best_dev_before") is None else round(o["dev"] - o["best_dev_before"], 2),
             "held-out change (mm)": None if o.get("best_heldout_before") is None else round(o["heldout"] - o["best_heldout_before"], 2)}
            for o in outs if o.get("accepted")]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Loss over time")
    c1, c2 = st.columns(2)
    logy = c1.toggle("log y-axis", value=True)
    strategies = sorted({o.get("strategy") for o in outs if o.get("strategy")})
    pick = c2.multiselect("strategies", strategies, default=strategies)
    s = state()
    best_iter = (s.get("best") or {}).get("iter_id")
    live_iter = s.get("iter_id") if str(s.get("phase", "")).startswith("training") else None
    fa = go.Figure()
    v0 = loss_log("0000")
    if v0:
        fa.add_trace(go.Scatter(x=[r["elapsed_s"] for r in v0], y=[r["loss"] for r in v0], name="v0",
                                line=dict(dash="dash", color=BLACK)))
    for o in outs:
        if o.get("strategy") not in pick:
            continue
        log = loss_log(o["iter_id"])
        if not log:
            continue
        is_best, is_live = o["iter_id"] == best_iter, o["iter_id"] == live_iter
        fa.add_trace(go.Scatter(
            x=[r["elapsed_s"] for r in log], y=[r["loss"] for r in log], name=o["iter_id"],
            line=dict(width=4 if is_best else 1.5, color=ORANGE if is_live else (GREEN if o.get("accepted") else RED)),
            opacity=1.0 if (is_best or is_live or o.get("accepted")) else 0.25))
    if live_iter and not any(o["iter_id"] == live_iter for o in outs):
        for label in ("heldout", "dev2", "dev"):
            log = loss_log(live_iter, label)
            if log:
                fa.add_trace(go.Scatter(x=[r["elapsed_s"] for r in log], y=[r["loss"] for r in log],
                                        name=f"{live_iter} live", line=dict(width=3, color=ORANGE)))
                break
    fa.update_layout(title="A: training loss vs elapsed seconds", xaxis_title="elapsed s", yaxis_title="loss",
                     yaxis_type="log" if logy else "linear", height=340, margin=dict(t=40))
    st.plotly_chart(fa, width="stretch")

    started = parse_ts(s.get("started_at", ""))
    ends_at = {e["iter_id"]: parse_ts(e["ts"]) for e in events() if e.get("type") == "iter_end" and e.get("iter_id")}
    fb = go.Figure()
    for kind, color in (("accepted", GREEN), ("rejected", RED), ("killed_early", GREY), ("failed", BLACK)):
        xs, ys, devs, sym, txt = [], [], [], [], []
        for o in outs:
            log = loss_log(o["iter_id"])
            t_end = ends_at.get(o["iter_id"])
            if status_of(o) != kind or not log or not t_end or not started:
                continue
            xs.append((t_end - started).total_seconds() / 60)
            ys.append(log[-1]["loss"])
            devs.append(o.get("dev"))
            sym.append("circle-open" if o.get("strategy") == "loss_and_target" else "circle")
            txt.append(o["iter_id"])
        if xs:
            fb.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=f"final loss ({kind})", text=txt,
                                    marker=dict(color=color, symbol=sym, size=10)))
            fb.add_trace(go.Scatter(x=xs, y=devs, mode="markers", name=f"dev ({kind})", yaxis="y2",
                                    marker=dict(color=color, symbol="diamond", size=7), opacity=0.6))
    fb.update_layout(title="B: final loss across the run (hollow = loss_and_target, different loss scale)",
                     xaxis_title="minutes since loop start", yaxis_title="final training loss",
                     yaxis2=dict(title="dev score (mm)", overlaying="y", side="right"), height=340, margin=dict(t=40))
    st.plotly_chart(fb, width="stretch")


# ---------- live run ----------
@st.fragment(run_every=5)
def live_tab():
    s = state()
    iter_id, phase = s.get("iter_id"), s.get("phase", "")
    label = {"training_dev": "dev", "training_dev_2": "dev2", "training_heldout": "heldout"}.get(phase)
    st.write(f"**{iter_id or '-'}** · {phase or '-'} · {s.get('strategy') or ''} — {s.get('rationale') or ''}")
    budget = CFG["train"]["time_budget_s"]
    el = s.get("train_elapsed_s") or 0
    st.progress(min(1.0, el / budget), text=f"training {el:.0f} / {budget} s, loss {s.get('train_loss')}")
    if iter_id and label:
        log = loss_log(iter_id, label)
        if log:
            fig = go.Figure(go.Scatter(x=[r["elapsed_s"] for r in log], y=[r["loss"] for r in log], mode="lines+markers"))
            fig.update_layout(xaxis_title="elapsed s", yaxis_title="loss", height=280, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
    evs = events()
    checks = [{"time": e["ts"], "elapsed_s": e.get("elapsed_s"), "answer": e.get("answer"), "fallback": e.get("used_fallback")}
              for e in evs if e.get("type") == "snap_check" and e.get("iter_id") == iter_id and e.get("run") == label]
    st.caption("Snap early-stop checks for this run")
    st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)
    st.caption("Event feed (last 30, newest first)")
    st.dataframe(pd.DataFrame(evs[-30:][::-1]), width="stretch", hide_index=True)


# ---------- attempts ----------
@st.fragment(run_every=5)
def attempts_tab():
    outs = attempts()
    if not outs:
        st.info("No attempts yet.")
        return
    df = pd.DataFrame([{"id": o["iter_id"], "strategy": o.get("strategy"), "rationale": o.get("rationale"),
                        "dev": o.get("dev"), "held-out": o.get("heldout"), "outcome": status_of(o),
                        "reason": o.get("reason"), "triage": o.get("triage"), "duration_s": o.get("duration_s")}
                       for o in outs])
    st.dataframe(df, width="stretch", hide_index=True)
    sel = st.selectbox("Show attempt", df["id"].tolist()[::-1], key="attempt_pick")
    if not sel:
        return
    diff = hfile(sel, "chosen.diff")
    if os.path.exists(diff):
        st.code(open(diff).read(), language="diff")
    cols = st.columns(3)
    for col, label in zip(cols, ("dev", "dev2", "heldout")):
        rep = rjson(hfile(sel, f"report_{label}.json"))
        if rep:
            col.caption(f"report_{label}.json")
            col.json(rep, expanded=False)
    for label in ("dev", "heldout"):
        if os.path.exists(hfile(sel, f"skeletons_{label}.png")):
            st.image(hfile(sel, f"skeletons_{label}.png"), caption=f"skeletons_{label}.png")
    fig = go.Figure()
    for label in ("dev", "dev2", "heldout"):
        log = loss_log(sel, label)
        if log:
            fig.add_trace(go.Scatter(x=[r["elapsed_s"] for r in log], y=[r["loss"] for r in log], name=label))
    if fig.data:
        fig.update_layout(xaxis_title="elapsed s", yaxis_title="loss", height=260, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    snapj = rjson(hfile(sel, "snap.json"))
    if snapj:
        st.caption("snap.json")
        st.dataframe(pd.DataFrame(snapj), width="stretch", hide_index=True)
    if os.path.exists(hfile(sel, "lesson.txt")):
        st.write("**Lesson:** " + open(hfile(sel, "lesson.txt")).read())


# ---------- skills ----------
@st.fragment(run_every=5)
def skills_tab():
    path = C.abspath(P["skills"])
    if not os.path.exists(path):
        st.info("No skills file.")
        return
    text = open(path).read()
    counts, sec = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            sec = line[3:].strip()
            counts[sec] = 0
        elif line.startswith("- ") and sec:
            counts[sec] += 1
    st.write(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))
    st.markdown(text)


# ---------- 3D viewer ----------
@st.cache_resource
def load_lifter_model(model_dir):
    from checker.check import load_model
    return load_model(model_dir)[0]


def tagged_models():
    return {o["tag"]: hfile(o["iter_id"], "model") for o in outcomes()
            if o.get("accepted") and o.get("tag") and os.path.exists(hfile(o["iter_id"], "model/model.pt"))}


def cam_rotation(az_deg, el_deg):
    """World->camera rotation used by datagen.project (aim jitter ignored): rows right, down, forward."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def skel_traces(p, name, color, width, opacity=1.0, legend=True):
    xs, ys, zs = [], [], []
    for a, b in BONES:
        xs += [p[a, 0], p[b, 0], None]
        ys += [p[a, 2], p[b, 2], None]
        zs += [-p[a, 1], -p[b, 1], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, showlegend=legend,
                        line=dict(color=color, width=width), opacity=opacity)


def viewer_tab():
    from checker.check import predict
    models = tagged_models()
    if "v0" not in models:
        st.info("No v0 model yet (milestone M4).")
        return
    latest = max(models, key=lambda t: int(t[1:]))
    c1, c2, c3 = st.columns(3)
    which = c1.radio("Eval set", ["dev", "heldout"], horizontal=True)
    d = np.load(C.abspath(P[which]))
    idx = c2.number_input("Sample index", 0, int(len(d["pose3d"])) - 1, 0)
    elev = c3.slider("Elevation (deg)", 30, 85, 45)
    az, _, dist, focal = d["cam"][idx]
    world = d["pose3d"][idx].astype(np.float64) @ cam_rotation(az, d["cam"][idx][1])   # camera -> world, z-up
    rng = np.random.default_rng(0)
    view = project(world[None], np.array([[az, elev, dist, focal]], dtype=np.float32), rng, CFG)
    gt = view["pose3d"][0].astype(np.float64)
    fig = go.Figure([skel_traces(gt, "ground truth", GREY, 8)])
    for tag, color in (("v0", RED), (latest, GREEN)):
        hyps = predict(load_lifter_model(models[tag]), view["kp2d"], view["mask"], CFG)[0]
        pred = hyps.mean(axis=0)
        st.write(f"**{tag}** MPJPE at {elev}°: {mpjpe_mm(pred[None], gt[None]):.1f} mm")
        fig.add_trace(skel_traces(pred, f"{tag} prediction", color, 5))
        if tag == latest:
            for k, h in enumerate(hyps[:10]):
                fig.add_trace(skel_traces(h, f"{tag} hypotheses", color, 1, 0.35, legend=k == 0))
    fig.update_layout(height=520, scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="depth", zaxis_title="up"),
                      margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")


# ---------- Snap and cost ----------
@st.fragment(run_every=5)
def snap_cost_tab():
    outs = attempts()
    if outs:
        df = pd.DataFrame([{"strategy": o.get("strategy") or "-", "accepted": bool(o.get("accepted"))} for o in outs])
        g = df.groupby("strategy").agg(attempts=("accepted", "size"), accepted=("accepted", "sum")).reset_index()
        g["acceptance_rate"] = g["accepted"] / g["attempts"]
        fig = go.Figure([go.Bar(x=g["strategy"], y=g["attempts"], name="attempts"),
                         go.Scatter(x=g["strategy"], y=g["acceptance_rate"], name="acceptance rate", yaxis="y2",
                                    mode="markers+lines")])
        fig.update_layout(yaxis_title="attempts", yaxis2=dict(title="acceptance rate", overlaying="y", side="right",
                                                              range=[0, 1]), height=300, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    calls = [c for o in outs for c in (rjson(hfile(o["iter_id"], "snap.json"), []) or [])]
    c = st.columns(4)
    if calls:
        fb = pd.DataFrame(calls).groupby("decision")["used_fallback"].mean()
        c[0].write("**Snap fallback rate**")
        c[0].dataframe(fb.rename("fallback rate").reset_index(), hide_index=True)
    evs = events()
    audits = [e for e in evs if e.get("type") == "snap_audit" and e.get("top_score") is not None]
    wins = [e for e in audits if e.get("bottom_score") is None or e["top_score"] < e["bottom_score"]]
    c[1].metric("Audit win rate", f"{len(wins) / len(audits):.0%}" if audits else "n/a", f"{len(audits)} audits",
                delta_color="off")
    saved = 0.0
    for o in outs:
        for label in ("dev", "dev2", "heldout"):
            rep = rjson(hfile(o["iter_id"], f"report_{label}.json"))
            if rep and rep.get("status") == "killed_early" and rep.get("train"):
                saved += CFG["train"]["time_budget_s"] - (rep["train"].get("elapsed_s") or 0)
    c[2].metric("Training minutes saved by early stops", f"{saved / 60:.1f}")
    llm = [e for e in evs if e.get("type") == "llm_call"]
    toks = sum((e.get("input_tokens") or 0) + (e.get("output_tokens") or 0) for e in llm)
    c[3].metric("Meta API", f"{len(llm)} calls", f"{toks:,} tokens, {np.mean([e['seconds'] for e in llm]) if llm else 0:.1f} s mean",
                delta_color="off")


# ---------- drone sim ----------
@st.fragment(run_every=5)
def sim_tab():
    frame, rows = C.abspath(P["sim_frame"]), rjsonl(C.abspath(P["sim_log"]))
    if not rows:
        st.info("No sim run yet: uv run python3 sim/run_sim.py")
        return
    c1, c2 = st.columns(2)
    if os.path.exists(frame):
        c1.image(frame, caption="live camera frame (mock Tello)")
    last = rows[-1]
    if last.get("gt"):
        fig = go.Figure([skel_traces(np.array(last["gt"]), "ground truth", GREY, 8)])
        if last.get("pred_v0"):
            fig.add_trace(skel_traces(np.array(last["pred_v0"]), "v0", RED, 5))
        fig.add_trace(skel_traces(np.array(last["pred_best"]), "best", GREEN, 5))
        fig.update_layout(height=380, scene=dict(aspectmode="data"), margin=dict(t=10))
        c2.plotly_chart(fig, width="stretch")
    df = pd.DataFrame(rows)
    path = pd.DataFrame(list(df["drone"]))
    top = go.Figure([go.Scatter(x=path["x"], y=path["y"], mode="lines+markers", name="drone"),
                     go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=14, symbol="star"), name="person")])
    top.update_layout(title="top-down path", xaxis_title="x (m)", yaxis_title="y (m)", height=360,
                      yaxis=dict(scaleanchor="x"), margin=dict(t=40))
    err = go.Figure([go.Scatter(x=df["t"], y=df["mpjpe_v0_mm"], name="v0", line=dict(color=RED)),
                     go.Scatter(x=df["t"], y=df["mpjpe_best_mm"], name="best", line=dict(color=GREEN)),
                     go.Scatter(x=df["t"], y=df["elevation_deg"], name="elevation", yaxis="y2", line=dict(dash="dot"))])
    err.update_layout(title="MPJPE over time", xaxis_title="t (s)", yaxis_title="MPJPE (mm)",
                      yaxis2=dict(title="elevation (deg)", overlaying="y", side="right"), height=360, margin=dict(t=40))
    c3, c4 = st.columns(2)
    c3.plotly_chart(top, width="stretch")
    c4.plotly_chart(err, width="stretch")


header()
tabs = st.tabs(["Progress", "Live run", "Attempts", "Skills", "3D viewer", "Snap and cost", "Drone sim"])
for tab, fn in zip(tabs, (progress_tab, live_tab, attempts_tab, skills_tab, viewer_tab, snap_cost_tab, sim_tab)):
    with tab:
        fn()
