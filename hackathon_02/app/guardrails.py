"""Deterministic guardrails.

LLM output is untrusted structured data. Anything that does not survive this
file is downgraded to a safe no_op instead of being pushed into the optimizer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ALLOWED = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _no_op(idx: int, why: str) -> Dict[str, Any]:
    return {
        "note_index": idx,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": why,
    }


def _clean_hours(raw: Any) -> Optional[List[int]]:
    if not isinstance(raw, list) or not raw:
        return None
    out = set()
    for h in raw:
        if isinstance(h, bool) or not isinstance(h, (int, float)):
            return None
        if float(h) != int(h) or not 0 <= int(h) <= 23:
            return None
        out.add(int(h))
    return sorted(out)


def _num(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    val = float(raw)
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return val


def validate(
    raw_items: List[Dict[str, Any]], notes: List[str], capacity_kwh: float
) -> List[Dict[str, Any]]:
    """Return exactly one clean entry per note, in note_index order."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idx = item.get("note_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if 0 <= idx < len(notes) and idx not in by_index:
            by_index[idx] = item

    result: List[Dict[str, Any]] = []
    for i in range(len(notes)):
        item = by_index.get(i)
        if item is None:
            result.append(_no_op(i, "No usable interpretation returned for this note."))
            continue

        dtype = item.get("directive_type")
        explanation = item.get("explanation")
        explanation = explanation if isinstance(explanation, str) and explanation else "Interpreted operator note."
        explanation = explanation[:300]

        if dtype not in ALLOWED or dtype == "no_op":
            result.append(_no_op(i, explanation if dtype == "no_op" else "Unsupported directive type; treated as no_op."))
            continue

        adj = item.get("structured_adjustment")
        if not isinstance(adj, dict):
            result.append(_no_op(i, "Missing structured adjustment; treated as no_op."))
            continue

        hours = _clean_hours(adj.get("hours"))
        if hours is None:
            result.append(_no_op(i, "Invalid hours; treated as no_op."))
            continue

        clean: Dict[str, Any] = {"hours": hours}

        if dtype == "solar_reduction":
            factor = _num(adj.get("factor"))
            if factor is None or not 0.0 <= factor <= 1.0:
                result.append(_no_op(i, "Invalid solar factor; treated as no_op."))
                continue
            clean["factor"] = factor
        elif dtype == "minimum_battery_reserve":
            reserve = _num(adj.get("minimum_energy_kwh"))
            if reserve is None or reserve < 0:
                result.append(_no_op(i, "Invalid reserve value; treated as no_op."))
                continue
            clean["minimum_energy_kwh"] = min(reserve, capacity_kwh)
        elif dtype == "max_grid_window":
            cap = _num(adj.get("max_grid_kwh"))
            if cap is None or cap < 0:
                result.append(_no_op(i, "Invalid grid cap; treated as no_op."))
                continue
            clean["max_grid_kwh"] = cap

        result.append(
            {
                "note_index": i,
                "applies": True,
                "directive_type": dtype,
                "structured_adjustment": clean,
                "explanation": explanation,
            }
        )
    return result


def build_constraints(directives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse validated directives into per-hour optimizer constraints."""
    solar_factor = {h: 1.0 for h in range(24)}
    reserve = {h: 0.0 for h in range(24)}
    no_charge, no_discharge = set(), set()
    grid_cap: Dict[int, float] = {}

    for d in directives:
        if not d["applies"]:
            continue
        adj, dtype = d["structured_adjustment"], d["directive_type"]
        for h in adj["hours"]:
            if dtype == "solar_reduction":
                solar_factor[h] = min(solar_factor[h], adj["factor"])
            elif dtype == "minimum_battery_reserve":
                reserve[h] = max(reserve[h], adj["minimum_energy_kwh"])
            elif dtype == "no_charge_window":
                no_charge.add(h)
            elif dtype == "no_discharge_window":
                no_discharge.add(h)
            elif dtype == "max_grid_window":
                grid_cap[h] = min(grid_cap.get(h, float("inf")), adj["max_grid_kwh"])

    return {
        "solar_factor": solar_factor,
        "reserve": reserve,
        "no_charge": no_charge,
        "no_discharge": no_discharge,
        "grid_cap": grid_cap,
    }
