"""LLM interpretation layer.

The generative model is the ONLY component allowed to read the raw operator
notes and decide what they mean. Everything it returns is treated as untrusted
data and is validated in guardrails.py before it can reach the optimizer.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
MODEL = os.getenv("LLM_MODEL", "")
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives.

Return ONLY a JSON object, no markdown fences, no prose:
{"directives":[{"note_index":0,"applies":true,"directive_type":"...","structured_adjustment":{...},"explanation":"..."}]}

Exactly one entry per note, in note_index order 0..N-1.

Allowed directive_type values and their required structured_adjustment:
- "solar_reduction"            -> {"hours":[...], "factor": number}
- "minimum_battery_reserve"    -> {"hours":[...], "minimum_energy_kwh": number}
- "no_charge_window"           -> {"hours":[...]}
- "no_discharge_window"        -> {"hours":[...]}
- "max_grid_window"            -> {"hours":[...], "max_grid_kwh": number}
- "no_op"                      -> null

Rules:
- "no_op" is used ONLY for notes that do not change the 24-hour energy schedule
  (menu changes, meetings, visitors, general announcements). no_op MUST have
  applies=false and structured_adjustment=null.
- Every other directive MUST have applies=true.
- hours: unique integers 0-23, ascending. Time windows are whole hours,
  START INCLUSIVE, END EXCLUSIVE. "1 PM to 3 PM" -> [13,14].
  "from 6 PM until 9 PM" -> [18,19,20]. "13:00-15:00" -> [13,14].
  "overnight 10 PM to 2 AM" -> [22,23,0,1] sorted ascending -> [0,1,22,23].
- solar_reduction factor is the FRACTION THAT REMAINS, between 0 and 1.
  "drops to about 20%" -> 0.2. "80% reduction" -> 0.2. "half the usual
  output" -> 0.5. "panels offline / no solar" -> 0.0.
- Never invent demand, tariff, battery parameters or a directive type that is
  not in the list. If a note is about energy but does not map to any allowed
  type, use no_op.
- explanation: one short sentence.
"""

USER_TEMPLATE = """Operator notes for one 24-hour scenario:

{notes}

Return the JSON object now."""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _call_gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = MODEL or "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = httpx.post(url, json=body, headers={"x-goog-api-key": key}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai_compatible(prompt: str) -> str:
    """Works for OpenAI, Groq, Together, OpenRouter, Ollama, vLLM, ..."""
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "not-needed")
    model = MODEL or "gpt-4o-mini"
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    r = httpx.post(
        f"{base}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    model = MODEL or "claude-sonnet-4-6"
    body = {
        "model": model,
        "max_tokens": 1500,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def interpret_notes(notes: List[str]) -> List[Dict[str, Any]]:
    """Ask the LLM to interpret every note. Raises on total failure."""
    numbered = "\n".join(f"[note_index {i}] {n}" for i, n in enumerate(notes))
    prompt = USER_TEMPLATE.format(notes=numbered)

    caller = {
        "gemini": _call_gemini,
        "openai": _call_openai_compatible,
        "groq": _call_openai_compatible,
        "openrouter": _call_openai_compatible,
        "local": _call_openai_compatible,
        "anthropic": _call_anthropic,
    }.get(PROVIDER, _call_gemini)

    last_error: Exception | None = None
    for _ in range(2):  # one retry; the model is the mandatory path
        try:
            raw = caller(prompt)
            data = _extract_json(raw)
            items = data.get("directives", data if isinstance(data, list) else [])
            if isinstance(items, list) and items:
                return items
            last_error = ValueError("empty directive list")
        except Exception as exc:  # noqa: BLE001 - never leak provider internals
            last_error = exc
    raise RuntimeError(f"LLM interpretation failed: {type(last_error).__name__}")
