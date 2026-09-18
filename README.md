# GridWise LLM Challenge — Campus Microgrid Optimizer

HTTP API that interprets natural-language operator notes with a language model,
validates the interpretation with deterministic guardrails, and solves a 24-hour
campus microgrid schedule that minimizes grid electricity cost in BDT.

## Architecture

```
request JSON
   -> LLM interpreter      (app/llm.py)         reads operator_notes, emits structured directives
   -> guardrail validator  (app/guardrails.py)  untrusted LLM output -> safe, typed constraints
   -> LP optimizer         (app/optimizer.py)   PuLP/CBC, min SUM(grid_kwh[h] * tariff[h])
   -> replay validator     (app/validator.py)   re-checks the plan the way the judge does
   -> response JSON        (app/main.py)
```

- **LLM role (mandatory step):** the model is the only component that reads the
  raw note text. It outputs `directive_type`, `hours`, and numeric values. It is
  never used only for `plan_summary`.
- **Guardrails:** directive type whitelist, one entry per note in `note_index`
  order, hours forced to unique ascending integers 0–23, `factor` clamped to
  [0,1], reserve clamped to battery capacity, non-negative grid cap,
  `applies=false` + `null` adjustment only for `no_op`. Anything that fails is
  downgraded to `no_op` instead of inventing a constraint.
- **Optimizer:** linear program over 24 hours with variables grid, solar used,
  charge, discharge, and battery state; constraints are hourly energy balance,
  battery bounds and rate limits, directive constraints, and end-of-day
  neutrality (`E[23] == initial_energy_kwh`).

## Quickstart (from a clean environment)

```bash
git clone <REPO_URL> && cd gridwise
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env         # then put your key in .env
export $(grep -v '^#' .env | xargs)

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

Public sample case:

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d @tests/sample_request.json | python3 -m json.tool
```

Offline pipeline test (no API key needed, LLM stubbed):

```bash
python3 tests/test_pipeline.py
```

## Configuration

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `gemini` (default), `openai`, `groq`, `openrouter`, `anthropic`, `local` |
| `LLM_MODEL` | model id, e.g. `gemini-2.0-flash`, `gpt-4o-mini`, `llama-3.3-70b-versatile` |
| `LLM_TIMEOUT_SECONDS` | per-call timeout, default `12` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | provider credential |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL (Groq, OpenRouter, Ollama, vLLM) |
| `PORT` | listen port, default `8000` |

No secret values are committed; `.env` is git-ignored and only variable *names*
are documented here.

## Docker fallback

```bash
docker build -t <USER>/gridwise:1.0.0 .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=gemini -e GEMINI_API_KEY=<key> \
  <USER>/gridwise:1.0.0
curl -s http://localhost:8000/health
```

The image binds to `0.0.0.0`, exposes `8000`, and contains no baked-in secrets.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | `200 {"status":"ok"}` |
| POST | `/optimize-energy` | `200` with `scenario_id`, `directive_interpretation`, `hourly_plan[24]`, `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`; `400` on malformed or structurally invalid request; `500` controlled internal error with no stack trace or secret |

## Failure handling

- LLM unreachable, slow, or returning malformed JSON: one retry, then every note
  is emitted as a guardrailed `no_op` and a physically valid schedule is still
  returned — the service never crashes and never invents a directive.
- If the directive set is infeasible together, the optimizer returns the base
  physically valid plan and says so in `plan_summary`.
- The plan is replayed before it is returned; if the replay finds any violation,
  a guaranteed-valid fallback plan is sent instead.

## Dependencies & credits

FastAPI, Uvicorn, Pydantic, httpx, PuLP (bundled CBC solver). LLM provider as
configured above.

## Known limitations

- Battery charge/discharge is modeled as lossless (no round-trip efficiency),
  matching the Problem Statement energy-balance equation.
- Grid export and inter-hour ramping limits are out of scope.
- Cost quality depends on the LP relaxation; simultaneous charge and discharge
  in the same hour is netted out before the plan is emitted.
