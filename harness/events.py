"""Event log (runs/events.jsonl) and state file (runs/state.json)."""
import json
import os
import tempfile
from datetime import datetime

from harness import config as C


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def emit(event_type: str, cfg: dict, **fields) -> None:
    """Append one JSON line with ts: {"ts", "type", "iter_id", ...fields}."""
    path = C.abspath(cfg["paths"]["events"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"ts": now_iso(), "type": event_type, **fields}
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read_state(cfg: dict) -> dict:
    path = C.abspath(cfg["paths"]["state"])
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def set_state(cfg: dict, **fields) -> None:
    """Merge fields into state.json and stamp last_update. Atomic write: tmp file, then rename."""
    path = C.abspath(cfg["paths"]["state"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = read_state(cfg)
    state.update(fields)
    state["last_update"] = now_iso()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".state_")
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)
