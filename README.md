# Flo

An autonomous research loop that improves a 3D human-pose lifter for a drone camera.

A conditional flow-matching model maps 2D body keypoints seen from a simulated DJI Tello-style camera to a 3D pose (17 COCO joints, meters). An LLM proposes edits to `candidate/train.py`. A checker trains each candidate for a fixed time budget and scores it on held-out data. A git-based gate decides whether the change becomes the new `main`.

demo: https://youtu.be/Vzut-d8jDBk
## How it works

Each iteration of `loop.py`:

1. **Route.** Pick a strategy from `config.yaml` (`tune_optimizer`, `timestep_sampling`, `architecture`, `conditioning`, `loss_and_target`, `regularization`, `fix_last_failure`).
2. **Propose.** `agents/coder.py` asks the model for 4 candidate edits (search/replace blocks against the current `train.py`).
3. **Rank.** `agents/snap.py` orders the candidates. Snap is currently disabled, so the original order is used.
4. **Check.** `checker/check.py` runs a static check, trains the candidate in a subprocess with a 180 s budget, samples hypotheses with an Euler ODE solver, and computes metrics.
5. **Gate.** `harness/gate.py` compares dev scores against the best so far, with a margin of 2 sigma. Held-out must not regress beyond 1 sigma.
6. **Record.** Each attempt writes `history/<iter>/` (outcome, diff, logs, lesson). Accepted attempts are merged to `main` and tagged.

Only `gate.accept()` moves `main`. Candidates live on `iter/<id>` branches until accepted.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and the H3WB pose files.

```bash
uv sync                                  # install dependencies
```

Create `.env` in the repo root with the model API key (the loop reads `MODEL_API_KEY`, or `META_API_KEY` as a fallback):

```
MODEL_API_KEY=...
```

Build the data and calibrate the gate (one-time):

```bash
uv run python3 checker/build_data.py     # writes data/train.npz, dev_eval.npz, heldout_eval.npz
uv run python3 checker/calibrate.py      # writes sigma into config.yaml, history/0000, tag v0
```

## Running

```bash
uv run python3 loop.py --hours 3 --no-snap          # the autonomous loop
uv run python3 loop.py --max-iters 1 --no-snap      # one iteration, for testing
```

Check a single candidate without the loop:

```bash
uv run python3 checker/check.py candidate --set dev --seed 0 --out /tmp/check
```

Dashboard (read-only JSON API plus a React UI):

```bash
cd dashboard/web && npm install && npm run build && cd -
uv run python3 dashboard/server.py --port 8502      # open http://127.0.0.1:8502
```

Simulated drone flight (orbit around a person, runs v0 and the chosen model on every frame):

```bash
uv run python3 sim/run_sim.py
```

Tests:

```bash
uv run pytest tests/
uv run python3 tests/run_adversarial.py             # adversarial candidates must fail with the right status
```

Final comparison with three seeds, written to `history/final.json`:

```bash
uv run python3 scripts/final_eval.py
```

## Metrics

Each candidate is scored on the dev set (seeds 0 and 1) and on a held-out set (a new subject from a higher camera).

- **MPJPE** (mm): error of the mean of the K samples. This is the main number.
- **PA-MPJPE** (mm): MPJPE after Procrustes alignment.
- **Best-of-K** (mm): error of the closest of the K samples.
- **Spread** (mm): disagreement between samples.
- **Bone error**: relative error on limb lengths. Above 0.30 the candidate is rejected as implausible.

Score = MPJPE + 50 × max(0, bone error − 0.15).

## Repository layout

| Path | What it is |
|---|---|
| `loop.py` | Autonomous loop entry point |
| `agents/` | LLM calls: `llm.py` (client), `coder.py` (proposes edits), `snap.py` (routing, ranking, triage; disabled) |
| `candidate/train.py` | The training script the loop edits (the model definition lives here) |
| `checker/` | Data generation (`datagen.py`, `build_data.py`), training check, sampler, metrics, calibration |
| `harness/` | `config.py` (loads `config.yaml`), `gate.py` (accept/reject), `history.py`, `events.py` |
| `config.yaml` | Data geometry, training budget, sampling, scoring, gate margins, loop settings |
| `skills/skills.md` | Lessons the coder reads: confirmed ideas and failed ideas |
| `history/` | Per-iteration records (`outcome.json`, `chosen.diff`, logs). `.pt` weights are ignored |
| `sim/` | Mock Tello, virtual world, and a flight script |
| `dashboard/` | `server.py` (API) and `web/` (React UI) |
| `scripts/` | `final_eval.py`, `compare_fmpose3d.py` (comparison with FMPose3D) |
| `tests/` | Unit tests and adversarial candidate checks |

## Notes

- **Time budget.** Training is limited by wall-clock time, so results depend on machine load. Avoid running other heavy jobs during a loop run.
- **Gate calibration.** `sigma_dev`, `sigma_heldout` and the baseline in `config.yaml` were calibrated at a 180 s budget. If you change the budget, recalibrate before trusting the gate.
- **Noise.** The same model scored about 0.4 mm apart on two dev seeds, so small gains are within noise. Multi-seed confirmation is the way to check borderline wins.
- **Status.** Accepted versions are tagged `v0`, `v1`, … in git. `git tag` lists them and `history/` has the numbers.
