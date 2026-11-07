"""Single loader for config.yaml. Every module gets its numbers from the dict returned here."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def abspath(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
