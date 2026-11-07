"""Iteration history under history/<iter_id>/. Coder-safe summaries never include held-out numbers."""
import json
import os

from harness import config as C


def _root(cfg: dict) -> str:
    return C.abspath(cfg["paths"]["history"])


def iter_ids(cfg: dict) -> list[str]:
    root = _root(cfg)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if d.isdigit() and len(d) == 4)


def new_iter_id(cfg: dict) -> str:
    """"0001", "0002", ... continuing from the highest existing id."""
    ids = [int(d) for d in iter_ids(cfg)]
    return f"{(max(ids) + 1) if ids else 1:04d}"


def save(iter_id: str, name: str, obj, cfg: dict) -> None:
    """dict/list -> json, str -> text."""
    path = os.path.join(_root(cfg), iter_id, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=2, default=str)


def load(iter_id: str, name: str, cfg: dict, default=None):
    path = os.path.join(_root(cfg), iter_id, name)
    try:
        with open(path) as f:
            return json.load(f) if name.endswith(".json") else f.read()
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def outcomes(cfg: dict) -> list[dict]:
    return [o for o in (load(d, "outcome.json", cfg) for d in iter_ids(cfg)) if o]


def summaries(n: int, cfg: dict) -> list[dict]:
    """Coder-safe: last n attempts (baseline excluded), dev score or status, no held-out numbers."""
    out = []
    for o in [o for o in outcomes(cfg) if o.get("iter_id") != "0000"][-n:]:
        result = "accepted" if o.get("accepted") else f"rejected: {o.get('reason')}"
        out.append({"id": o["iter_id"], "strategy": o.get("strategy"), "rationale": o.get("rationale"),
                    "dev": o.get("dev") if o.get("dev") is not None else o.get("reason"), "result": result})
    return out


def best(cfg: dict) -> dict:
    """{tag, dev, heldout, iter_id} of the current main: the last accepted attempt, else the v0 baseline."""
    chosen = {}
    for o in outcomes(cfg):
        if o.get("accepted"):
            chosen = o
    return {"tag": chosen.get("tag"), "dev": chosen.get("dev"), "heldout": chosen.get("heldout"),
            "iter_id": chosen.get("iter_id")}
