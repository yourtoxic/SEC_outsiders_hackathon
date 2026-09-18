"""Offline check: runs the full pipeline with a stubbed LLM (no API key needed).

    python3 tests/test_pipeline.py
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = json.load(open(os.path.join(HERE, "sample_request.json")))

STUB = [
    {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
     "explanation": "Solar drops to 20% for two hours."},
    {"note_index": 1, "applies": True, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]},
     "explanation": "No charging in that window."},
    {"note_index": 2, "applies": False, "directive_type": "no_op",
     "structured_adjustment": None,
     "explanation": "Does not affect the schedule."},
]

client = TestClient(app, raise_server_exceptions=False)
failures = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        failures.append(name)


r = client.get("/health")
check("/health returns 200 ok", r.status_code == 200 and r.json() == {"status": "ok"})

with patch("app.llm.interpret_notes", return_value=STUB):
    r = client.post("/optimize-energy", json=REQ)
body = r.json()
check("/optimize-energy returns 200", r.status_code == 200)
check("scenario_id echoed", body["scenario_id"] == REQ["scenario_id"])
check("one interpretation per note",
      [d["note_index"] for d in body["directive_interpretation"]] == [0, 1, 2])
check("no_op uses applies=false + null",
      body["directive_interpretation"][2]["applies"] is False
      and body["directive_interpretation"][2]["structured_adjustment"] is None)
check("24 hourly plan entries",
      [p["hour"] for p in body["hourly_plan"]] == list(range(24)))
check("no charging in hours 14-15",
      all(not (p["hour"] in (14, 15) and p["battery_action"] == "charge")
          for p in body["hourly_plan"]))
check("solar capped at 20% in hours 13-14",
      all(p["solar_used_kwh"] <= REQ["hours"][p["hour"]]["solar_kwh"] * 0.2 + 0.01
          for p in body["hourly_plan"] if p["hour"] in (13, 14)))
check("end-of-day neutrality",
      abs(body["hourly_plan"][23]["battery_energy_after_kwh"]
          - REQ["battery"]["initial_energy_kwh"]) <= 0.01)
check("totals match hourly_plan",
      abs(sum(p["grid_kwh"] for p in body["hourly_plan"]) - body["total_grid_kwh"]) <= 0.01)

check("malformed JSON -> 400",
      client.post("/optimize-energy", content="{oops",
                  headers={"content-type": "application/json"}).status_code == 400)
check("invalid schema -> 400",
      client.post("/optimize-energy", json={"scenario_id": "x"}).status_code == 400)

with patch("app.llm.interpret_notes", side_effect=RuntimeError("provider down")):
    check("LLM outage still returns a valid 200",
          client.post("/optimize-energy", json=REQ).status_code == 200)

print("\n" + (f"{len(failures)} FAILED" if failures else "all checks passed"))
sys.exit(1 if failures else 0)
