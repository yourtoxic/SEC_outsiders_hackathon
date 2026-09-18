"""GridWise LLM Challenge — API service.

Pipeline:  request -> LLM interpretation -> deterministic guardrails
           -> LP optimizer -> final replay validation -> JSON response
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import guardrails, llm, optimizer, validator
from .schemas import OptimizeRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM Challenge", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError):
    # Spec asks for 400 on structurally invalid requests.
    return JSONResponse(status_code=400, content={"error": "invalid request schema"})


@app.exception_handler(Exception)
async def _server_error(_: Request, exc: Exception):
    log.error("unhandled error: %s", type(exc).__name__)  # never log secrets/traces
    return JSONResponse(status_code=500, content={"error": "internal error"})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy")
def optimize_energy(req: OptimizeRequest) -> JSONResponse:
    notes = [n.strip() for n in req.operator_notes]
    if any(not n for n in notes):
        return JSONResponse(status_code=400, content={"error": "empty operator note"})

    hours_by_id = {h.hour: h.model_dump() for h in req.hours}
    if sorted(hours_by_id) != list(range(24)):
        return JSONResponse(status_code=400, content={"error": "hours must cover 0..23 exactly once"})
    hours: List[Dict[str, float]] = [hours_by_id[h] for h in range(24)]
    battery = req.battery.model_dump()

    # 1) LLM interpretation (mandatory path)
    llm_ok = True
    try:
        raw_items: List[Dict[str, Any]] = llm.interpret_notes(notes)
    except Exception as exc:  # noqa: BLE001
        log.warning("llm failure: %s", type(exc).__name__)
        raw_items, llm_ok = [], False

    # 2) Deterministic guardrails — anything unsafe becomes no_op
    directives = guardrails.validate(raw_items, notes, battery["capacity_kwh"])
    cons = guardrails.build_constraints(directives)

    # 3) Optimization
    plan, applied = optimizer.optimize(hours, battery, cons)

    # 4) Final replay — if our own plan fails, fall back to a guaranteed-valid one
    issues = validator.replay(plan, hours, battery, cons)
    if issues:
        log.warning("replay issues: %s", issues[:5])
        plan, applied = optimizer.optimize(hours, battery, guardrails.build_constraints([]))

    total_grid, total_cost, peak = optimizer.totals(plan, hours)
    active = [d["directive_type"] for d in directives if d["applies"]]
    summary = (
        f"Applied {len(active)} operator directive(s)"
        + (f" ({', '.join(sorted(set(active)))})" if active else "")
        + f"; met all 24 hours of demand with solar first, shifted battery charging to "
        f"cheap-tariff hours and discharging to expensive hours, returned the battery to "
        f"its initial state of charge, and bought {total_grid:.2f} kWh from the grid for "
        f"{total_cost:.2f} BDT."
    )
    if not llm_ok:
        summary += " Note interpretation ran in degraded mode."
    if not applied and active:
        summary += " Directive set was infeasible together; base physical plan returned."

    return JSONResponse(
        status_code=200,
        content={
            "scenario_id": req.scenario_id,
            "directive_interpretation": directives,
            "hourly_plan": plan,
            "total_grid_kwh": total_grid,
            "total_cost_bdt": total_cost,
            "peak_grid_kwh": peak,
            "plan_summary": summary,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
