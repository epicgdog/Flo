"""Flight script: fly an orbit around a person and run v0 and the chosen model on every frame.

CLI: uv run python3 sim/run_sim.py [--real] [--model-tag v3]
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checker.datagen import load_h3wb_body  # noqa: E402
from checker.metrics import mpjpe_mm  # noqa: E402
from harness import config as C  # noqa: E402
from sim.keypoints import detector_keypoints, sim_keypoints  # noqa: E402
from sim.lifter import Lifter  # noqa: E402
from sim.world import World  # noqa: E402


def goto(tello, tx: float, ty: float, tz: float) -> None:
    """Fly to a world point (meters) with go_xyz_speed (body frame, cm)."""
    s = tello.sim_state()
    dx, dy, dz = tx - s["x"], ty - s["y"], tz - s["z"]
    yaw = math.radians(s["yaw"])
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    tello.go_xyz_speed(int(round(fwd * 100)), int(round(left * 100)), int(round(dz * 100)), 50)


def face_origin(tello) -> None:
    s = tello.sim_state()
    delta = (math.degrees(math.atan2(-s["y"], -s["x"])) - tello.get_yaw() + 180.0) % 360.0 - 180.0
    if delta > 0:
        tello.rotate_clockwise(int(round(delta)))
    elif delta < 0:
        tello.rotate_counter_clockwise(int(round(-delta)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="fly a real DJI Tello (needs detector_keypoints)")
    ap.add_argument("--model-tag", default="latest")
    args = ap.parse_args()
    cfg = C.load()
    sim = cfg["sim"]
    cfg_sim_real = args.real
    if cfg_sim_real:
        from djitellopy import Tello
    else:
        from sim.mock_tello import MockTello as Tello

    v0 = Lifter("v0", cfg)
    best = Lifter(args.model_tag, cfg)
    if cfg_sim_real:
        tello = Tello()
        world = None
    else:
        world = World(load_h3wb_body(C.abspath(cfg["paths"]["raw_test_json"]), cfg), cfg,
                      seed=cfg["data"]["seeds"]["heldout"])
        tello = Tello(world=world, cfg=cfg)

    log_path = C.abspath(cfg["paths"]["sim_log"])
    frame_path = C.abspath(cfg["paths"]["sim_frame"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    errs0, errsb = [], []
    t0 = time.time()
    tello.connect()
    tello.streamon()
    tello.takeoff()
    try:
        with open(log_path, "w") as log:
            for h_cm in sim["orbit_heights_cm"]:
                for r in sim["orbit_radius_m"]:
                    for i in range(sim["orbit_steps"]):
                        th = 2 * math.pi * i / sim["orbit_steps"]
                        goto(tello, r * math.cos(th), r * math.sin(th), h_cm / 100.0)
                        face_origin(tello)
                        if cfg_sim_real:
                            frame = tello.get_frame_read().frame
                            kp, m = detector_keypoints(frame)
                            gt = None
                        else:
                            world_out = world.render(tello.sim_state(), None)
                            frame, gt = world_out[0], world_out[3].astype(np.float64)
                            kp, m = sim_keypoints(world_out, cfg)
                        p0, _ = v0.lift(kp, m)
                        pb, _ = best.lift(kp, m)
                        e0 = mpjpe_mm(p0[None], gt[None]) if gt is not None else None
                        eb = mpjpe_mm(pb[None], gt[None]) if gt is not None else None
                        if gt is not None:
                            errs0.append(e0)
                            errsb.append(eb)
                        s = tello.sim_state() if not cfg_sim_real else {}
                        horiz = math.hypot(s.get("x", 0), s.get("y", 0))
                        rec = {"t": round(time.time() - t0, 2), "drone": s,
                               "elevation_deg": round(math.degrees(math.atan2(s.get("z", 0), horiz)), 2),
                               "distance_m": round(math.hypot(horiz, s.get("z", 0)), 3), "n_visible": int(m.sum()),
                               "mpjpe_v0_mm": e0, "mpjpe_best_mm": eb, "kp2d": kp.tolist(),
                               "pred_best": pb.tolist(), "pred_v0": p0.tolist(),
                               "gt": gt.tolist() if gt is not None else None}
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        tello.land()
        tello.end()
    print(f"frames: {len(errs0)}")
    if errs0:
        print(f"mean mpjpe_v0_mm: {np.mean(errs0):.1f}   mean mpjpe_best_mm ({best.tag}): {np.mean(errsb):.1f}")
    print(f"wrote {log_path} and {frame_path}")


if __name__ == "__main__":
    main()
