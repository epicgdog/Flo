"""Meta Model API via the openai package's Responses endpoint. The only place that calls the coder model."""
import os
import time

import openai

from harness import config as C
from harness import events

RETRIES = 3
ENV_FILE = os.path.join(C.ROOT, ".env")
TOKENS = {"input": 0, "output": 0}   # running totals; the loop reads them for outcome.json llm_tokens


def _load_dotenv() -> None:
    """Minimal .env loader (uv run does not load .env). Existing environment wins.
    The project .env names the key META_API_KEY; the spec reads MODEL_API_KEY, so it is mapped."""
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("MODEL_API_KEY") and os.environ.get("META_API_KEY"):
        os.environ["MODEL_API_KEY"] = os.environ["META_API_KEY"]


def chat(model: str, system: str, user: str, cfg: dict,
         max_output_tokens: int | None = None, effort: str | None = None) -> str:
    m = cfg["meta"]
    _load_dotenv()
    client = openai.OpenAI(base_url=m["base_url"], api_key=os.environ["MODEL_API_KEY"], timeout=m["timeout_s"])
    effort = effort or m["coder_effort"]
    max_output_tokens = max_output_tokens or m["max_output_tokens"]
    last_exc = None
    for attempt in range(RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.responses.create(model=model, instructions=system, input=user,
                                           reasoning={"effort": effort}, max_output_tokens=max_output_tokens)
        except Exception as e:  # any API error, including HTTP 429
            last_exc = e
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
            continue
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "input_tokens", None) or 0
        tout = getattr(usage, "output_tokens", None) or 0
        TOKENS["input"] += tin
        TOKENS["output"] += tout
        events.emit("llm_call", cfg, model=model, input_tokens=tin, output_tokens=tout,
                    seconds=round(time.time() - t0, 2))
        return resp.output_text
    raise RuntimeError(f"Meta API failed after {RETRIES} retries: {last_exc!r}")
